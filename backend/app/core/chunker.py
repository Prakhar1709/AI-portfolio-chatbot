import re
from pathlib import Path
from typing import List, Dict, Any

class MarkdownChunk:
    def __init__(
        self,
        chunk_id: str,
        content: str,
        source_file: str,
        category: str,
        section_title: str,
        breadcrumb: str,
        tags: List[str] = None,
        char_count: int = 0
    ):
        self.chunk_id = chunk_id
        self.content = content.strip()
        self.source_file = source_file
        self.category = category
        self.section_title = section_title
        self.breadcrumb = breadcrumb
        self.tags = tags or []
        self.char_count = char_count or len(self.content)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source_file": self.source_file,
            "category": self.category,
            "section_title": self.section_title,
            "breadcrumb": self.breadcrumb,
            "tags": ",".join(self.tags),
            "char_count": self.char_count
        }

class StructureAwareMarkdownChunker:
    """
    Structure-aware recursive markdown chunker that preserves header hierarchy,
    breadcrumbs, and metadata.
    """
    def __init__(self, max_chunk_size: int = 700, chunk_overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_file(self, file_path: Path) -> List[MarkdownChunk]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        file_name = file_path.name
        category = file_path.stem.lower()

        return self.chunk_markdown_text(text, file_name=file_name, category=category)

    def chunk_markdown_text(self, text: str, file_name: str, category: str) -> List[MarkdownChunk]:
        lines = text.split("\n")
        sections = []
        
        current_headers = {}  # level -> title
        current_section_lines = []
        current_section_title = "Introduction"
        
        def get_current_breadcrumb():
            # Build breadcrumb like "Projects > OmniRAG > Architecture"
            sorted_levels = sorted(current_headers.keys())
            if not sorted_levels:
                return current_section_title
            return " > ".join([current_headers[lvl] for lvl in sorted_levels])

        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.*)$', line.strip())
            if header_match:
                # Flush previous section if there's content
                if current_section_lines:
                    content_str = "\n".join(current_section_lines).strip()
                    if content_str:
                        sections.append({
                            "title": current_section_title,
                            "breadcrumb": get_current_breadcrumb(),
                            "content": content_str
                        })
                    current_section_lines = []

                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # Clear deeper header levels
                keys_to_remove = [k for k in current_headers.keys() if k >= level]
                for k in keys_to_remove:
                    del current_headers[k]
                
                current_headers[level] = title
                current_section_title = title
            else:
                current_section_lines.append(line)

        # Flush final section
        if current_section_lines:
            content_str = "\n".join(current_section_lines).strip()
            if content_str:
                sections.append({
                    "title": current_section_title,
                    "breadcrumb": get_current_breadcrumb(),
                    "content": content_str
                })

        # Process sections into final chunks (splitting large sections recursively)
        chunks: List[MarkdownChunk] = []
        chunk_counter = 0

        for sec in sections:
            sec_content = sec["content"]
            breadcrumb = sec["breadcrumb"]
            title = sec["title"]
            
            # Extract basic tags from bold terms or lists in the section
            tags = self._extract_tags(sec_content, title)

            if len(sec_content) <= self.max_chunk_size:
                chunk_id = f"{category}_{self._slugify(title)}_{chunk_counter}"
                chunk_counter += 1
                
                # Contextualized text with breadcrumb header
                formatted_content = f"[{breadcrumb}]\n{sec_content}"
                chunks.append(
                    MarkdownChunk(
                        chunk_id=chunk_id,
                        content=formatted_content,
                        source_file=file_name,
                        category=category,
                        section_title=title,
                        breadcrumb=breadcrumb,
                        tags=tags,
                        char_count=len(formatted_content)
                    )
                )
            else:
                # Recursive split by paragraphs or sentences
                sub_chunks = self._split_large_text(sec_content, self.max_chunk_size, self.chunk_overlap)
                for sub_idx, sub_text in enumerate(sub_chunks):
                    chunk_id = f"{category}_{self._slugify(title)}_{chunk_counter}_{sub_idx}"
                    formatted_content = f"[{breadcrumb} (Part {sub_idx+1})]\n{sub_text}"
                    chunks.append(
                        MarkdownChunk(
                            chunk_id=chunk_id,
                            content=formatted_content,
                            source_file=file_name,
                            category=category,
                            section_title=title,
                            breadcrumb=breadcrumb,
                            tags=tags,
                            char_count=len(formatted_content)
                        )
                    )
                chunk_counter += 1

        return chunks

    def _split_large_text(self, text: str, max_size: int, overlap: int) -> List[str]:
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_len = 0

        for p in paragraphs:
            p_len = len(p)
            if current_len + p_len > max_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                # Keep last paragraph as overlap if reasonable
                if p_len < max_size:
                    current_chunk = [p]
                    current_len = p_len
                else:
                    # Paragraph itself is too large, split by sentences/lines
                    current_chunk = []
                    current_len = 0
                    lines = p.split("\n")
                    temp = []
                    temp_len = 0
                    for l in lines:
                        if temp_len + len(l) > max_size and temp:
                            chunks.append("\n".join(temp))
                            temp = [l]
                            temp_len = len(l)
                        else:
                            temp.append(l)
                            temp_len += len(l) + 1
                    if temp:
                        current_chunk = temp
                        current_len = temp_len
            else:
                current_chunk.append(p)
                current_len += p_len + 2

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _extract_tags(self, content: str, title: str) -> List[str]:
        tags = set()
        # Find tech stack or bold tags
        bold_matches = re.findall(r'\*\*(.*?)\*\*', content)
        for b in bold_matches[:5]:
            clean_b = re.sub(r'[^\w\s-]', '', b).strip()
            if 2 < len(clean_b) < 25:
                tags.add(clean_b)
        
        # Add title keywords
        for w in title.split():
            if len(w) > 4:
                tags.add(w.lower())
        return list(tags)[:6]

    def _slugify(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-z0-9]+', '_', text).strip('_')
        return text[:25] or "sec"
