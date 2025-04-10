"""
API routes for answering questions based on document search results
"""

import logging
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Query, status, Depends

from app.api.dependencies import get_splade_service
from app.core.config import settings
from app.core.search_utils import deduplicate_and_threshold_results, merge_chunk_content
from app.core.splade_service import SpladeService
from app.llm.client import AsyncLLMClient

router = APIRouter()
logger = logging.getLogger(__name__)


async def synthesize_results_with_llm(
        query: str,
        search_results: List[Dict[str, Any]],
        model: str = None
) -> str:
    """
    Use LLM to synthesize information from search results into a coherent answer
    
    Args:
        query: User's original question
        search_results: List of search results
        model: LLM model to use (defaults to settings.LLM_DEFAULT_MODEL)
        
    Returns:
        Synthesized answer as a string
    """
    if not search_results:
        return "No relevant information was found to answer your question."

    # Use default model if none provided
    model_to_use = model or settings.LLM_DEFAULT_MODEL or "gpt-4o-mini"

    # Create LLM client
    llm_client = AsyncLLMClient()

    # Prepare context from search results (limit to top results to avoid token limits)
    top_results = search_results[:5]
    contexts = []

    for i, result in enumerate(top_results):
        # Format result with metadata
        metadata_str = ""
        if result.get("metadata"):
            # Filter out technical metadata
            filtered_metadata = {k: v for k, v in result["metadata"].items()
                                 if k not in ["embedding", "vector", "chunk_vector"]}
            if filtered_metadata:
                metadata_str = f"\nMetadata: {filtered_metadata}"

        contexts.append(f"Document {i + 1} (score: {result['score']:.2f}){metadata_str}\n{result['content']}")

    combined_context = "\n\n---\n\n".join(contexts)

    prompt = f"""
    Based on these document chunks, answer this question: "{query}"
    
    Document information:
    {combined_context}
    
    Provide a comprehensive answer that synthesizes information across all relevant chunks.
    If the information is not available in the provided chunks, state this clearly.
    """

    try:
        response = await llm_client.response(
            user_input=prompt,
            model=model_to_use,
            temperature=0.7,
            max_tokens=1000
        )

        return response.get("text", "Failed to generate an answer.")
    except Exception as e:
        # Return the error
        logger.error(f"Error synthesizing results: {e}")
        return f"Error generating answer: {str(e)}"


@router.get("/{collection_id}")
async def answer_query(
        collection_id: str,
        question: str = Query(..., description="User question"),
        model: Optional[str] = Query(None, description="Optional LLM model to use"),
        top_k: int = Query(10, description="Number of search results to consider"),
        min_score: float = Query(0.3, description="Minimum similarity score threshold (0-1)"),
        query_expansion: bool = Query(True, description="Use LLM to expand the search query"),
        splade_service: SpladeService = Depends(get_splade_service)
):
    """
    Answer a user question using search results and LLM synthesis
    
    - Takes a natural language question
    - Performs a search (with optional query expansion)
    - Uses an LLM to synthesize the results into a comprehensive answer
    """
    # Check if LLM integration is enabled
    if not settings.LLM_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="LLM integration is disabled"
        )

    # Check if collection exists
    if not splade_service.get_collection(collection_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection with ID {collection_id} not found"
        )

    # Use the advanced search with query expansion
    # We're reusing the query expansion logic from advanced_search
    search_query = question
    expanded_query = None

    # Import here to avoid circular import
    from app.api.routes.advanced_search import expand_query_with_llm

    if query_expansion:
        try:
            expanded = await expand_query_with_llm(question, model)
            if expanded and expanded != question:
                logger.info(f"Query expanded from '{question}' to '{expanded}'")
                search_query = expanded
                expanded_query = expanded
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")

    # Perform the search
    results, query_time = splade_service.search(
        collection_id,
        search_query,
        top_k,
        None  # No metadata filter for simplicity
    )

    # Apply post-processing
    if min_score > 0:
        results = deduplicate_and_threshold_results(
            results,
            min_score_threshold=min_score,
            deduplicate_by_original_id=True
        )

    # Merge chunks for better context
    has_chunks = any(r.get("metadata", {}).get("is_chunk", False) for r in results)
    if has_chunks:
        results = merge_chunk_content(results)

    # Synthesize answer from search results
    answer = await synthesize_results_with_llm(question, results, model)

    # Return response with answer and search results
    response = {
        "question": question,
        "answer": answer,
        "search_results": results,
        "query_time_ms": query_time
    }

    # Add expanded query if used
    if expanded_query:
        response["expanded_query"] = expanded_query

    return response


@router.get("/")
@router.get("")
async def answer_query_all_collections(
        question: str = Query(..., description="User question"),
        model: Optional[str] = Query(None, description="Optional LLM model to use"),
        top_k: int = Query(5, description="Number of search results per collection"),
        min_score: float = Query(0.3, description="Minimum similarity score threshold (0-1)"),
        query_expansion: bool = Query(True, description="Use LLM to expand the search query"),
        splade_service: SpladeService = Depends(get_splade_service)
):
    """
    Answer a user question by searching across all collections
    
    - Searches all collections for relevant information
    - Synthesizes a comprehensive answer using an LLM
    """
    # Check if LLM integration is enabled
    if not settings.LLM_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="LLM integration is disabled"
        )

    # Use the query expansion logic from advanced_search
    search_query = question
    expanded_query = None

    # Import here to avoid circular import
    from app.api.routes.advanced_search import expand_query_with_llm

    if query_expansion:
        try:
            expanded = await expand_query_with_llm(question, model)
            if expanded and expanded != question:
                logger.info(f"Query expanded from '{question}' to '{expanded}'")
                search_query = expanded
                expanded_query = expanded
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")

    # Search across all collections
    search_results = splade_service.search_all_collections(
        search_query,
        top_k,
        None,  # No metadata filter
        None,  # No geo filter
        min_score
    )

    # Get results from all collections
    all_results = []
    for collection_id, collection_results in search_results["results"].items():
        # Post-process each collection's results
        processed_results = deduplicate_and_threshold_results(
            collection_results,
            min_score_threshold=min_score,
            deduplicate_by_original_id=True
        )

        # Merge chunks if needed
        has_chunks = any(r.get("metadata", {}).get("is_chunk", False) for r in processed_results)
        if has_chunks:
            processed_results = merge_chunk_content(processed_results)

        # Add collection ID to each result for reference
        for result in processed_results:
            result["collection_id"] = collection_id

        all_results.extend(processed_results)

    # Sort all results by score
    sorted_results = sorted(all_results, key=lambda x: x["score"], reverse=True)

    # Take top results across all collections
    top_results = sorted_results[:top_k * 2]  # Get more results for better synthesis

    # Synthesize answer from search results
    answer = await synthesize_results_with_llm(question, top_results, model)

    # Return response with answer and search results
    response = {
        "question": question,
        "answer": answer,
        "search_results": top_results,
        "query_time_ms": search_results["query_time_ms"]
    }

    # Add expanded query if used
    if expanded_query:
        response["expanded_query"] = expanded_query

    return response
