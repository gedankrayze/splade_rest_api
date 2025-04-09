# Handling Large Documents & Tables

This guide explains how the SPLADE Content Server handles extremely large documents (100+ pages) and large tables.

## Challenges with Large Documents

Processing very large documents introduces several challenges:

1. **Token Limits**: Language models have fixed token limits, usually 512 tokens for SPLADE
2. **Context Preservation**: Important to maintain context when splitting documents
3. **Table Integrity**: Tables need special handling to preserve their structure
4. **Performance Concerns**: Processing huge documents can be resource-intensive

## Enhanced Large Document Processing

The SPLADE Content Server includes special features for handling large documents:

### Hierarchical Document Segmentation

For very large documents (100+ pages), the system uses a hierarchical approach:

1. **Document → Sections → Chunks**:
    - First divides the document into major sections (by headings)
    - Then processes each section into optimally sized chunks
    - Maintains section metadata across chunks

2. **Semantic Structure Preservation**:
    - Respects document hierarchy (headings, subheadings)
    - Adds section context to each chunk's metadata
    - Ensures logical boundaries between chunks

### Large Table Handling

Tables receive special treatment to ensure their usefulness in search results:

1. **Table Preservation**:
    - Tables smaller than the configured limit stay intact
    - Tables are treated as semantic units and kept together when possible

2. **Large Table Splitting**:
    - For tables exceeding size limits (e.g., 100+ rows), intelligent splitting occurs
    - Table headers are repeated in each chunk for context
    - A note is added to indicate the table has been split

3. **Table Structure Recognition**:
    - Automatically identifies Markdown table syntax
    - Preserves table formatting and alignment
    - Handles complex tables with varied column counts

### Configuration Options

The system offers configurable settings to optimize for your specific document types:

```
# In app/core/config.py or .env file
SPLADE_MAX_CHUNK_SIZE=500      # Regular content chunk size (in tokens)
SPLADE_TABLE_CHUNK_SIZE=1000   # Larger limit for table chunks
SPLADE_CHUNK_OVERLAP=50        # Overlap between chunks
```

## Practical Examples

### Handling a 140-Page Technical Document

Consider a large technical manual with multiple sections and tables:

1. **Document Processing**:
    - The document is first segmented into major sections by headings
    - Each section maintains its context through metadata
    - Section content is further chunked as needed

2. **Table Treatment**:
    - Small tables (<50 rows) remain intact within their chunks
    - Medium tables (50-100 rows) use a larger chunk size allowance
    - Very large tables (>100 rows) are split with header repetition

3. **Search Experience**:
    - Users can search using section-specific terms
    - Table content is properly represented in search results
    - Advanced search can merge related chunks from the same section

### Example Document Structure

```
Original Document (140 pages)
│
├── Section 1: Introduction
│   ├── Chunk 1.1: Overview text
│   └── Chunk 1.2: More introduction with small table
│
├── Section 2: Technical Specifications
│   ├── Chunk 2.1: Text and small table
│   ├── Chunk 2.2: Medium-sized table (preserved intact)
│   └── Chunk 2.3: More specifications
│
└── Section 3: Detailed Analysis
    ├── Chunk 3.1: Analysis introduction
    ├── Chunk 3.2: Large table (part 1 with headers)
    └── Chunk 3.3: Large table (part 2 with repeated headers)
```

## Performance Considerations

The enhanced chunking system is optimized for both small and large documents:

1. **Progressive Processing**:
    - Smaller documents (<20 pages) use standard chunking
    - Medium documents (20-100 pages) use enhanced paragraph handling
    - Very large documents (100+ pages) use full hierarchical segmentation

2. **Memory Efficiency**:
    - Processes one section at a time to manage memory usage
    - Avoids loading the entire document into memory at once

3. **Search Performance**:
    - Fast retrieval even with large document collections
    - Metadata-enhanced search to target specific sections

## Best Practices

When working with large documents:

1. **Document Structure**: Use clear heading levels (# for main sections, ## for subsections)
2. **Table Design**: Keep tables focused and well-structured when possible
3. **Section Organization**: Group related content into logical sections
4. **Metadata Enrichment**: Add metadata to identify document type, format, and purpose
5. **Chunk Size Tuning**: Adjust chunk sizes based on your document characteristics

## Technical Implementation

The system uses a multi-stage approach to processing:

1. **Content Type Detection**: Identifies document format (Markdown, plain text)
2. **Document Segmentation**: Breaks large documents into manageable sections
3. **Semantic Block Extraction**: Identifies special elements like tables and code blocks
4. **Adaptive Chunking**: Uses different strategies based on content type
5. **Chunk Optimization**: Ensures optimal chunk size with proper context overlap

By leveraging this enhanced processing, the SPLADE Content Server effectively handles documents of virtually any size
while maintaining search quality and performance.
