#!/bin/bash

# Script to upload markdown documents to SPLADE Content Server
# Usage: ./upload_docs.sh <folder_path> <collection_id> <api_url>
# Example: ./upload_docs.sh ./docs technical-docs http://localhost:8000

# Ensure jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed. Please install jq."
    echo "You can install it via: "
    echo "  - macOS: brew install jq"
    echo "  - Ubuntu/Debian: sudo apt-get install jq"
    echo "  - CentOS/RHEL: sudo yum install jq"
    exit 1
fi

# Check if curl is installed
if ! command -v curl &> /dev/null; then
    echo "Error: curl is required but not installed. Please install curl."
    exit 1
fi

# Validate arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <folder_path> <collection_id> [api_url]"
    echo "Example: $0 ./docs technical-docs http://localhost:8000"
    exit 1
fi

FOLDER_PATH="$1"
COLLECTION_ID="$2"
API_URL="${3:-http://localhost:8000}"

# Check if the folder exists
if [ ! -d "$FOLDER_PATH" ]; then
    echo "Error: Folder $FOLDER_PATH does not exist."
    exit 1
fi

# Check if the collection exists, create if not
collection_check=$(curl -s "$API_URL/collections/$COLLECTION_ID")
if echo "$collection_check" | grep -q "not found"; then
    echo "Collection $COLLECTION_ID does not exist. Creating..."
    collection_data="{\"id\": \"$COLLECTION_ID\", \"name\": \"$COLLECTION_ID\", \"description\": \"Uploaded documents\"}"
    create_result=$(curl -s -X POST "$API_URL/collections" \
         -H "Content-Type: application/json" \
         -d "$collection_data")

    if echo "$create_result" | grep -q "error"; then
        echo "Error creating collection: $create_result"
        exit 1
    else
        echo "Collection created successfully."
    fi
else
    echo "Collection $COLLECTION_ID already exists."
fi

# Function to upload a single document
upload_document() {
    local file="$1"
    local filename=$(basename "$file")
    local doc_id="${filename%.*}"  # Remove extension to use as document ID

    # Read file content
    local content=$(cat "$file")

    # Create JSON payload
    local json_data=$(jq -n \
        --arg id "$doc_id" \
        --arg content "$content" \
        '{id: $id, content: $content, metadata: {source: "uploaded", filename: $id}}')

    echo "Uploading $filename as document ID: $doc_id"

    # Upload document using curl
    local result=$(curl -s -X POST "$API_URL/documents/$COLLECTION_ID" \
         -H "Content-Type: application/json" \
         -d "$json_data")

    # Check for errors
    if echo "$result" | grep -q "error"; then
        echo "Error uploading $filename: $result"
        return 1
    else
        echo "Successfully uploaded $filename"
        return 0
    fi
}

# Find all markdown files and upload them
echo "Starting upload of markdown documents from $FOLDER_PATH to collection $COLLECTION_ID at $API_URL"
echo "================================================="

TOTAL_FILES=0
SUCCESSFUL=0
FAILED=0

find "$FOLDER_PATH" -type f -name "*.md" | while read file; do
    TOTAL_FILES=$((TOTAL_FILES + 1))
    if upload_document "$file"; then
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

echo "================================================="
echo "Upload summary:"
echo "Total files processed: $TOTAL_FILES"
echo "Successfully uploaded: $SUCCESSFUL"
echo "Failed uploads: $FAILED"

# Test search functionality
if [ $SUCCESSFUL -gt 0 ]; then
    echo ""
    echo "Testing search functionality:"
    echo "================================================="

    # Extract a random term from one of the uploaded documents for searching
    SEARCH_TERM=$(grep -o -w "\w\{5,\}" "$FOLDER_PATH"/*.md | sort -R | head -1 | cut -d':' -f2)

    echo "Performing basic search with term: $SEARCH_TERM"
    search_result=$(curl -s "$API_URL/search/$COLLECTION_ID?query=$SEARCH_TERM")
    result_count=$(echo "$search_result" | jq '.results | length')
    echo "Found $result_count results for basic search"

    echo ""
    echo "Performing advanced search with query expansion:"
    advanced_result=$(curl -s "$API_URL/advanced-search/$COLLECTION_ID?query=$SEARCH_TERM&query_expansion=true")

    # Check if query was expanded
    if echo "$advanced_result" | jq -e '.expanded_query' > /dev/null; then
        expanded_query=$(echo "$advanced_result" | jq -r '.expanded_query')
        echo "Query expanded to: $expanded_query"
    else
        echo "Query was not expanded."
    fi

    adv_result_count=$(echo "$advanced_result" | jq '.results | length')
    echo "Found $adv_result_count results for advanced search"

    echo ""
    echo "Testing question answering:"
    question="What is the main topic of these documents?"
    echo "Question: $question"

    query_result=$(curl -s "$API_URL/query/$COLLECTION_ID?question=$(echo $question | jq -sRr @uri)")
    answer=$(echo "$query_result" | jq -r '.answer')

    echo "Answer: $answer"
fi

echo ""
echo "Script completed. You can now test more complex queries using:"
echo "  - Basic search: curl \"$API_URL/search/$COLLECTION_ID?query=your+query\""
echo "  - Advanced search: curl \"$API_URL/advanced-search/$COLLECTION_ID?query=your+query&query_expansion=true\""
echo "  - Question answering: curl \"$API_URL/query/$COLLECTION_ID?question=your+question\""