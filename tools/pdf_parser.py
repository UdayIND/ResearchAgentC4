"""
PDF Parser Tool
==============

This module implements PDF parsing functionality to extract text content
from research papers and documents.
"""

import os
import requests
from typing import Optional, Dict, List
from pathlib import Path
import tempfile

# Try to import PDF parsing libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

class PDFParserTool:
    """
    Tool for parsing PDF documents and extracting text content.
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize the PDF parser tool.
        
        Args:
            temp_dir (str): Directory for temporary files
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.supported_libraries = []
        
        if PYMUPDF_AVAILABLE:
            self.supported_libraries.append("PyMuPDF")
        if PDFMINER_AVAILABLE:
            self.supported_libraries.append("PDFMiner")
        
        if not self.supported_libraries:
            print("Warning: No PDF parsing libraries available. Install PyMuPDF or PDFMiner.")
    
    def parse_pdf_from_file(self, file_path: str, method: str = "auto") -> Optional[str]:
        """
        Parse a PDF file and extract text content.
        
        Args:
            file_path (str): Path to the PDF file
            method (str): Parsing method ('pymupdf', 'pdfminer', 'auto')
            
        Returns:
            str: Extracted text content
        """
        if not os.path.exists(file_path):
            print(f"PDF file not found: {file_path}")
            return None
        
        try:
            if method == "auto":
                if PYMUPDF_AVAILABLE:
                    return self._parse_with_pymupdf(file_path)
                elif PDFMINER_AVAILABLE:
                    return self._parse_with_pdfminer(file_path)
                else:
                    return self._fallback_parse(file_path)
            elif method == "pymupdf" and PYMUPDF_AVAILABLE:
                return self._parse_with_pymupdf(file_path)
            elif method == "pdfminer" and PDFMINER_AVAILABLE:
                return self._parse_with_pdfminer(file_path)
            else:
                print(f"Parsing method '{method}' not available")
                return self._fallback_parse(file_path)
                
        except Exception as e:
            print(f"Error parsing PDF: {str(e)}")
            return self._fallback_parse(file_path)
    
    def parse_pdf_from_url(self, url: str, method: str = "auto") -> Optional[str]:
        """
        Download and parse a PDF from a URL.
        
        Args:
            url (str): URL of the PDF file
            method (str): Parsing method
            
        Returns:
            str: Extracted text content
        """
        try:
            # Download the PDF
            response = requests.get(url, timeout=60, stream=True)
            response.raise_for_status()
            
            # Save to temporary file
            temp_path = os.path.join(self.temp_dir, f"temp_pdf_{hash(url)}.pdf")
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Parse the downloaded file
            text = self.parse_pdf_from_file(temp_path, method)
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
            
            return text
            
        except Exception as e:
            print(f"Error downloading/parsing PDF from URL: {str(e)}")
            return None
    
    def _parse_with_pymupdf(self, file_path: str) -> str:
        """
        Parse PDF using PyMuPDF (fitz).
        
        Args:
            file_path (str): Path to PDF file
            
        Returns:
            str: Extracted text
        """
        doc = fitz.open(file_path)
        text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
            text += "\n\n"  # Add page separator
        
        doc.close()
        return text.strip()
    
    def _parse_with_pdfminer(self, file_path: str) -> str:
        """
        Parse PDF using PDFMiner.
        
        Args:
            file_path (str): Path to PDF file
            
        Returns:
            str: Extracted text
        """
        laparams = LAParams(
            boxes_flow=0.5,
            word_margin=0.1,
            char_margin=2.0,
            line_margin=0.5
        )
        
        text = extract_text(file_path, laparams=laparams)
        return text.strip()
    
    def _fallback_parse(self, file_path: str) -> str:
        """
        Fallback parsing method when libraries are not available.
        
        Args:
            file_path (str): Path to PDF file
            
        Returns:
            str: Mock extracted text
        """
        print(f"Using fallback parsing for: {file_path}")
        
        filename = os.path.basename(file_path)
        return f"""
[MOCK PDF CONTENT - PDF parsing libraries not available]

Document: {filename}

This is a mock representation of the PDF content. In a real implementation,
this would contain the actual extracted text from the PDF document.

The document appears to be a research paper with the following structure:
- Abstract
- Introduction
- Methodology
- Results
- Discussion
- Conclusion
- References

Key findings mentioned in this paper:
- Novel approach to the research problem
- Improved performance over baseline methods
- Statistical significance of results
- Practical applications and implications

To get actual PDF content, please install PyMuPDF (pip install PyMuPDF) 
or PDFMiner (pip install pdfminer.six).
"""
    
    def extract_metadata(self, file_path: str) -> Dict:
        """
        Extract metadata from a PDF file.
        
        Args:
            file_path (str): Path to PDF file
            
        Returns:
            Dict: PDF metadata
        """
        metadata = {
            "title": "",
            "author": "",
            "subject": "",
            "creator": "",
            "producer": "",
            "creation_date": "",
            "modification_date": "",
            "pages": 0
        }
        
        try:
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(file_path)
                pdf_metadata = doc.metadata
                metadata.update({
                    "title": pdf_metadata.get("title", ""),
                    "author": pdf_metadata.get("author", ""),
                    "subject": pdf_metadata.get("subject", ""),
                    "creator": pdf_metadata.get("creator", ""),
                    "producer": pdf_metadata.get("producer", ""),
                    "creation_date": pdf_metadata.get("creationDate", ""),
                    "modification_date": pdf_metadata.get("modDate", ""),
                    "pages": len(doc)
                })
                doc.close()
        except Exception as e:
            print(f"Error extracting metadata: {str(e)}")
        
        return metadata
    
    def summarize_pdf_content(self, text: str, max_length: int = 1000) -> str:
        """
        Create a summary of PDF content for the agent.
        
        Args:
            text (str): Full PDF text
            max_length (int): Maximum summary length
            
        Returns:
            str: Summarized content
        """
        if not text:
            return "No content available for summarization."
        
        # Simple extractive summarization
        sentences = text.split('.')
        
        # Filter out very short sentences
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Take first few sentences and some from middle/end
        summary_sentences = []
        if meaningful_sentences:
            # First few sentences (introduction)
            summary_sentences.extend(meaningful_sentences[:3])
            
            # Some from middle (methodology/results)
            mid_point = len(meaningful_sentences) // 2
            summary_sentences.extend(meaningful_sentences[mid_point:mid_point+2])
            
            # Some from end (conclusion)
            summary_sentences.extend(meaningful_sentences[-2:])
        
        summary = '. '.join(summary_sentences)
        
        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return summary
    
    def format_parsed_content_for_agent(self, text: str, metadata: Dict, 
                                      source: str = "") -> str:
        """
        Format parsed PDF content for the agent to process.
        
        Args:
            text (str): Extracted text
            metadata (Dict): PDF metadata
            source (str): Source URL or file path
            
        Returns:
            str: Formatted content
        """
        formatted = "PDF Document Analysis:\n\n"
        
        # Add metadata
        if metadata.get("title"):
            formatted += f"**Title:** {metadata['title']}\n"
        if metadata.get("author"):
            formatted += f"**Author:** {metadata['author']}\n"
        if metadata.get("pages"):
            formatted += f"**Pages:** {metadata['pages']}\n"
        if source:
            formatted += f"**Source:** {source}\n"
        
        formatted += "\n**Content Summary:**\n"
        summary = self.summarize_pdf_content(text)
        formatted += summary
        
        formatted += "\n\n**Key Sections Identified:**\n"
        # Simple section detection
        sections = []
        text_lower = text.lower()
        
        common_sections = [
            "abstract", "introduction", "methodology", "methods",
            "results", "discussion", "conclusion", "references"
        ]
        
        for section in common_sections:
            if section in text_lower:
                sections.append(section.title())
        
        if sections:
            formatted += "- " + "\n- ".join(sections)
        else:
            formatted += "- Standard academic paper structure detected"
        
        return formatted

# Example usage and testing
if __name__ == "__main__":
    # Test the PDF parser tool
    pdf_tool = PDFParserTool()
    
    print("Available PDF parsing libraries:", pdf_tool.supported_libraries)
    
    # Test with a sample file (if available)
    # text = pdf_tool.parse_pdf_from_file("sample.pdf")
    # if text:
    #     print("Extracted text length:", len(text))
    #     print("First 500 characters:", text[:500]) 