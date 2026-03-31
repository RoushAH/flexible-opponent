"""Text extraction from PDFs, images, and plain text files."""

import asyncio
import io
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from ..llm.client import LLMClient


@dataclass
class ExtractedImage:
    """An image extracted from a document."""

    page_number: int
    image_index: int  # Index within the page
    image_bytes: bytes
    description: str  # AI-generated description
    source_file: str


@dataclass
class ExtractedPage:
    """A page of extracted text with metadata."""

    page_number: int
    text: str
    source_file: str
    source_type: str  # "pdf", "image", "text"
    images: list[ExtractedImage] = field(default_factory=list)


@dataclass
class ExtractedDocument:
    """A complete extracted document."""

    pages: list[ExtractedPage]
    source_file: str
    total_pages: int
    images: list[ExtractedImage] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Get the full text of all pages, including image descriptions."""
        parts = []
        for page in self.pages:
            parts.append(page.text)
            # Add image descriptions
            for img in page.images:
                parts.append(f"\n[DIAGRAM on page {img.page_number}]: {img.description}\n")
        return "\n\n".join(parts)


def extract_from_pdf(file_path: Path) -> tuple[ExtractedDocument, list[tuple[int, int, bytes]]]:
    """Extract text and images from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Tuple of (ExtractedDocument with text, list of (page_num, img_index, image_bytes)).
    """
    reader = PdfReader(file_path)
    pages = []
    raw_images = []  # (page_number, image_index, bytes)

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(
            ExtractedPage(
                page_number=i + 1,
                text=text.strip(),
                source_file=str(file_path),
                source_type="pdf",
            )
        )

        # Extract images from this page
        if hasattr(page, "images"):
            for img_idx, image in enumerate(page.images):
                try:
                    raw_images.append((i + 1, img_idx, image.data))
                except Exception:
                    # Skip images that can't be extracted
                    pass

    return (
        ExtractedDocument(
            pages=pages,
            source_file=str(file_path),
            total_pages=len(pages),
        ),
        raw_images,
    )


async def analyze_pdf_image(
    image_bytes: bytes,
    page_number: int,
    image_index: int,
    client: LLMClient,
    source_file: str,
) -> ExtractedImage:
    """Analyze a single image from a PDF using Claude vision.

    Args:
        image_bytes: Raw image bytes.
        page_number: Page the image was found on.
        image_index: Index of image on the page.
        client: LLM client for vision.
        source_file: Source PDF path.

    Returns:
        ExtractedImage with description.
    """
    response = await client.complete_with_images(
        system_prompt="You are analyzing diagrams and images from a board game rulebook. Describe what you see in detail, focusing on game-relevant information like board layout, piece positions, resource tracks, card layouts, and setup instructions.",
        user_prompt="""Describe this diagram/image from a board game rulebook. Focus on:
1. What type of diagram is it? (board layout, setup example, component reference, gameplay example)
2. What game elements are shown? (spaces, pieces, cards, tokens, tracks)
3. Any spatial relationships or positions that matter for gameplay
4. Any numbers, labels, or text visible in the diagram

Be specific and detailed - this description will be used to understand the game rules.""",
        images=[image_bytes],
        temperature=0.2,
    )

    return ExtractedImage(
        page_number=page_number,
        image_index=image_index,
        image_bytes=image_bytes,
        description=response.content.strip(),
        source_file=source_file,
    )


async def extract_from_pdf_with_images(
    file_path: Path,
    client: LLMClient,
    max_images: int = 20,
    min_image_size: int = 5000,  # Skip tiny images (icons, etc.)
) -> ExtractedDocument:
    """Extract text and analyze images from a PDF file.

    Args:
        file_path: Path to the PDF file.
        client: LLM client for image analysis.
        max_images: Maximum number of images to analyze (to control costs).
        min_image_size: Minimum image size in bytes to analyze.

    Returns:
        ExtractedDocument with text and image descriptions.
    """
    # First extract text and raw images
    document, raw_images = extract_from_pdf(file_path)

    # Filter and limit images
    filtered_images = [
        (page, idx, data)
        for page, idx, data in raw_images
        if len(data) >= min_image_size
    ][:max_images]

    if not filtered_images:
        return document

    print(f"    Analyzing {len(filtered_images)} images from PDF...")

    # Analyze images concurrently
    tasks = [
        analyze_pdf_image(data, page, idx, client, str(file_path))
        for page, idx, data in filtered_images
    ]
    analyzed_images = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out any failed analyses
    valid_images = [img for img in analyzed_images if isinstance(img, ExtractedImage)]

    # Attach images to their respective pages
    for img in valid_images:
        for page in document.pages:
            if page.page_number == img.page_number:
                page.images.append(img)
                break

    # Also store all images at document level
    document.images = valid_images

    return document


def extract_from_text(file_path: Path) -> ExtractedDocument:
    """Extract text from a plain text file.

    Args:
        file_path: Path to the text file.

    Returns:
        ExtractedDocument with the text content.
    """
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    return ExtractedDocument(
        pages=[
            ExtractedPage(
                page_number=1,
                text=text.strip(),
                source_file=str(file_path),
                source_type="text",
            )
        ],
        source_file=str(file_path),
        total_pages=1,
    )


async def extract_from_image(
    file_path: Path,
    client: LLMClient,
) -> ExtractedDocument:
    """Extract text from an image using Claude vision.

    Args:
        file_path: Path to the image file.
        client: LLM client for vision API.

    Returns:
        ExtractedDocument with extracted text.
    """
    # Read image bytes
    with open(file_path, "rb") as f:
        image_bytes = f.read()

    # Use Claude vision to extract text
    response = await client.complete_with_images(
        system_prompt="You are a precise text extractor. Extract all text from this image exactly as written, preserving formatting and structure. Include headers, bullet points, and any visible text.",
        user_prompt="Extract all text from this rulebook page. Preserve the structure (headers, sections, bullet points). Return only the extracted text, no commentary.",
        images=[image_bytes],
        temperature=0.1,  # Low temperature for accurate extraction
    )

    return ExtractedDocument(
        pages=[
            ExtractedPage(
                page_number=1,
                text=response.content.strip(),
                source_file=str(file_path),
                source_type="image",
            )
        ],
        source_file=str(file_path),
        total_pages=1,
    )


async def extract_from_images(
    file_paths: list[Path],
    client: LLMClient,
) -> ExtractedDocument:
    """Extract text from multiple images (e.g., scanned rulebook pages).

    Args:
        file_paths: Paths to image files, in order.
        client: LLM client for vision API.

    Returns:
        ExtractedDocument with extracted text from all images.
    """
    # Process images concurrently
    tasks = [extract_from_image(path, client) for path in file_paths]
    results = await asyncio.gather(*tasks)

    # Combine into single document
    pages = []
    for i, doc in enumerate(results):
        for page in doc.pages:
            pages.append(
                ExtractedPage(
                    page_number=i + 1,
                    text=page.text,
                    source_file=page.source_file,
                    source_type="image",
                )
            )

    return ExtractedDocument(
        pages=pages,
        source_file=str(file_paths[0].parent),
        total_pages=len(pages),
    )


async def extract_document(
    source: Path | str | list[Path],
    client: LLMClient | None = None,
    extract_pdf_images: bool = True,
) -> ExtractedDocument:
    """Extract text from any supported source.

    Args:
        source: Path to file, raw text string, or list of image paths.
        client: LLM client (required for image extraction/analysis).
        extract_pdf_images: If True and client provided, analyze images in PDFs.

    Returns:
        ExtractedDocument with extracted text (and image descriptions if applicable).

    Raises:
        ValueError: If source type is unsupported or client missing for images.
    """
    # Handle raw text string
    if isinstance(source, str) and not Path(source).exists():
        return ExtractedDocument(
            pages=[
                ExtractedPage(
                    page_number=1,
                    text=source.strip(),
                    source_file="<raw_text>",
                    source_type="text",
                )
            ],
            source_file="<raw_text>",
            total_pages=1,
        )

    # Handle list of image paths
    if isinstance(source, list):
        if client is None:
            raise ValueError("LLM client required for image extraction")
        return await extract_from_images(source, client)

    # Handle single file path
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        # If client provided, extract and analyze images too
        if client is not None and extract_pdf_images:
            return await extract_from_pdf_with_images(path, client)
        else:
            document, _ = extract_from_pdf(path)
            return document

    elif suffix == ".txt" or suffix == ".md":
        return extract_from_text(path)

    elif suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        if client is None:
            raise ValueError("LLM client required for image extraction")
        return await extract_from_image(path, client)

    else:
        raise ValueError(f"Unsupported file type: {suffix}")
