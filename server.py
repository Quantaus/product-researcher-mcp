"""
Product Researcher MCP Server
==============================
An MCP server that gives Claude Code the ability to research any product,
tool, API, or service by searching the web and returning structured briefs.

Search-API-agnostic: supports Tavily, Brave Search, and Serper.
Just set the right env vars and go.
"""

import os
import json
import httpx
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
from mcp.server.fastmcp import FastMCP

# --- Server Init ---

mcp = FastMCP("product_researcher_mcp")

# --- Config ---

SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "tavily")  # tavily | brave | serper
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

DEFAULT_MAX_RESULTS = 8
REQUEST_TIMEOUT = 30.0

# --- Search Adapters ---

async def _search_tavily(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> List[Dict[str, Any]]:
    """Search using Tavily API (best for AI agents)."""
    if not TAVILY_API_KEY:
        return [{"error": "TAVILY_API_KEY not set. Get one free at https://tavily.com"}]

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "include_answer": True,
                "include_raw_content": False,
                "search_depth": "advanced",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    if data.get("answer"):
        results.append({"title": "AI Summary", "content": data["answer"], "url": ""})
    for r in data.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "content": r.get("content", ""),
            "url": r.get("url", ""),
        })
    return results


async def _search_brave(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> List[Dict[str, Any]]:
    """Search using Brave Search API."""
    if not BRAVE_API_KEY:
        return [{"error": "BRAVE_API_KEY not set. Get one free at https://brave.com/search/api/"}]

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": BRAVE_API_KEY, "Accept": "application/json"},
            params={"q": query, "count": max_results},
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for r in data.get("web", {}).get("results", []):
        results.append({
            "title": r.get("title", ""),
            "content": r.get("description", ""),
            "url": r.get("url", ""),
        })
    return results


async def _search_serper(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> List[Dict[str, Any]]:
    """Search using Serper API (Google results)."""
    if not SERPER_API_KEY:
        return [{"error": "SERPER_API_KEY not set. Get one at https://serper.dev"}]

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": max_results},
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    if data.get("answerBox"):
        box = data["answerBox"]
        results.append({
            "title": "Featured Answer",
            "content": box.get("snippet", box.get("answer", "")),
            "url": box.get("link", ""),
        })
    for r in data.get("organic", []):
        results.append({
            "title": r.get("title", ""),
            "content": r.get("snippet", ""),
            "url": r.get("link", ""),
        })
    return results


async def search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> List[Dict[str, Any]]:
    """Route to the configured search provider."""
    providers = {
        "tavily": _search_tavily,
        "brave": _search_brave,
        "serper": _search_serper,
    }
    provider_fn = providers.get(SEARCH_PROVIDER.lower())
    if not provider_fn:
        return [{"error": f"Unknown SEARCH_PROVIDER '{SEARCH_PROVIDER}'. Use: tavily, brave, or serper"}]

    try:
        return await provider_fn(query, max_results)
    except httpx.HTTPStatusError as e:
        return [{"error": f"Search API returned {e.response.status_code}: {e.response.text[:200]}"}]
    except httpx.TimeoutException:
        return [{"error": "Search request timed out. Try again."}]
    except Exception as e:
        return [{"error": f"Search failed: {type(e).__name__}: {str(e)[:200]}"}]


def _format_results_markdown(results: List[Dict[str, Any]]) -> str:
    """Format search results as markdown."""
    if not results:
        return "No results found."
    if results[0].get("error"):
        return f"**Error:** {results[0]['error']}"

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        content = r.get("content", "No content")
        url = r.get("url", "")
        lines.append(f"### {i}. {title}")
        lines.append(content)
        if url:
            lines.append(f"Source: {url}")
        lines.append("")
    return "\n".join(lines)


# --- Input Models ---

class ResearchProductInput(BaseModel):
    """Input for researching a product/tool/service."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    product_name: str = Field(
        ...,
        description="Name of the product, tool, API, or service to research (e.g., 'Kling AI', 'Supabase', 'Stripe')",
        min_length=1,
        max_length=200,
    )
    focus: Optional[str] = Field(
        default=None,
        description="Optional focus area: 'overview', 'pricing', 'api', 'alternatives', 'technical', or a custom question",
        max_length=300,
    )


class CompareProductsInput(BaseModel):
    """Input for comparing two or more products."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    products: List[str] = Field(
        ...,
        description="List of product names to compare (e.g., ['Supabase', 'Firebase', 'PlanetScale'])",
        min_length=2,
        max_length=5,
    )
    criteria: Optional[str] = Field(
        default=None,
        description="What to compare on (e.g., 'pricing', 'developer experience', 'scalability')",
        max_length=300,
    )


class SearchWebInput(BaseModel):
    """Input for a raw web search."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        ...,
        description="Search query to run",
        min_length=1,
        max_length=400,
    )
    max_results: int = Field(
        default=DEFAULT_MAX_RESULTS,
        description="Maximum number of results to return",
        ge=1,
        le=20,
    )


class LookupPricingInput(BaseModel):
    """Input for looking up product pricing."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    product_name: str = Field(
        ...,
        description="Product name to look up pricing for",
        min_length=1,
        max_length=200,
    )


class FindAlternativesInput(BaseModel):
    """Input for finding alternatives to a product."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    product_name: str = Field(
        ...,
        description="Product to find alternatives for",
        min_length=1,
        max_length=200,
    )
    use_case: Optional[str] = Field(
        default=None,
        description="Specific use case to find alternatives for (e.g., 'video generation', 'database for serverless')",
        max_length=300,
    )


# --- Tools ---

@mcp.tool(
    name="research_product",
    annotations={
        "title": "Research a Product",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def research_product(params: ResearchProductInput) -> str:
    """Research any product, tool, API, or service. Returns a comprehensive brief
    covering what it does, how it works, key features, pricing, and technical details.

    Args:
        params (ResearchProductInput): Contains:
            - product_name (str): The product to research
            - focus (Optional[str]): Specific area to focus on

    Returns:
        str: Markdown-formatted product research brief
    """
    product = params.product_name
    focus = params.focus

    queries = []
    if focus == "pricing":
        queries = [f"{product} pricing plans 2025 2026", f"{product} free tier cost"]
    elif focus == "api":
        queries = [f"{product} API documentation developer", f"{product} SDK integration guide"]
    elif focus == "alternatives":
        queries = [f"{product} alternatives competitors comparison", f"best {product} alternatives 2026"]
    elif focus == "technical":
        queries = [f"{product} architecture how it works technical", f"{product} tech stack infrastructure"]
    elif focus:
        queries = [f"{product} {focus}", f"{product} {focus} details"]
    else:
        queries = [
            f"{product} what is features overview",
            f"{product} pricing plans",
            f"{product} API developer integration",
        ]

    all_results = []
    for q in queries:
        results = await search(q, max_results=5)
        all_results.extend(results)

    seen_urls = set()
    unique_results = []
    for r in all_results:
        url = r.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)

    output_lines = [
        f"# Product Research: {product}",
        "",
    ]
    if focus:
        output_lines.append(f"**Focus:** {focus}")
        output_lines.append("")

    output_lines.append(f"**Search provider:** {SEARCH_PROVIDER}")
    output_lines.append(f"**Queries run:** {len(queries)}")
    output_lines.append(f"**Results found:** {len(unique_results)}")
    output_lines.append("")
    output_lines.append("---")
    output_lines.append("")
    output_lines.append(_format_results_markdown(unique_results))

    return "\n".join(output_lines)


@mcp.tool(
    name="compare_products",
    annotations={
        "title": "Compare Products",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def compare_products(params: CompareProductsInput) -> str:
    """Compare two or more products side by side. Searches for each product
    and their direct comparisons to build a comparison brief.

    Args:
        params (CompareProductsInput): Contains:
            - products (List[str]): Products to compare
            - criteria (Optional[str]): Comparison criteria

    Returns:
        str: Markdown-formatted comparison brief
    """
    products = params.products
    criteria = params.criteria or "features pricing developer experience"

    all_results = {}

    for product in products:
        results = await search(f"{product} {criteria}", max_results=4)
        all_results[product] = results

    if len(products) == 2:
        vs_query = f"{products[0]} vs {products[1]} {criteria}"
        vs_results = await search(vs_query, max_results=4)
        all_results["Head-to-Head"] = vs_results
    elif len(products) > 2:
        vs_query = " vs ".join(products[:3]) + f" comparison {criteria}"
        vs_results = await search(vs_query, max_results=4)
        all_results["Comparison"] = vs_results

    output_lines = [
        f"# Product Comparison: {' vs '.join(products)}",
        "",
        f"**Criteria:** {criteria}",
        f"**Search provider:** {SEARCH_PROVIDER}",
        "",
        "---",
        "",
    ]

    for section, results in all_results.items():
        output_lines.append(f"## {section}")
        output_lines.append("")
        output_lines.append(_format_results_markdown(results))
        output_lines.append("")

    return "\n".join(output_lines)


@mcp.tool(
    name="lookup_pricing",
    annotations={
        "title": "Look Up Product Pricing",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def lookup_pricing(params: LookupPricingInput) -> str:
    """Look up pricing and plans for a specific product or service.

    Args:
        params (LookupPricingInput): Contains:
            - product_name (str): Product to look up pricing for

    Returns:
        str: Markdown-formatted pricing information
    """
    product = params.product_name

    results = await search(f"{product} pricing plans cost 2026", max_results=6)

    output_lines = [
        f"# Pricing: {product}",
        "",
        f"**Search provider:** {SEARCH_PROVIDER}",
        "",
        "---",
        "",
        _format_results_markdown(results),
    ]

    return "\n".join(output_lines)


@mcp.tool(
    name="find_alternatives",
    annotations={
        "title": "Find Product Alternatives",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def find_alternatives(params: FindAlternativesInput) -> str:
    """Find alternatives and competitors to a given product or service.

    Args:
        params (FindAlternativesInput): Contains:
            - product_name (str): Product to find alternatives for
            - use_case (Optional[str]): Specific use case context

    Returns:
        str: Markdown-formatted list of alternatives with details
    """
    product = params.product_name
    use_case = params.use_case

    queries = [f"best {product} alternatives 2026"]
    if use_case:
        queries.append(f"best tools for {use_case} alternatives to {product}")
    queries.append(f"{product} competitors comparison")

    all_results = []
    for q in queries:
        results = await search(q, max_results=5)
        all_results.extend(results)

    seen_urls = set()
    unique_results = []
    for r in all_results:
        url = r.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)

    output_lines = [
        f"# Alternatives to {product}",
        "",
    ]
    if use_case:
        output_lines.append(f"**Use case:** {use_case}")
        output_lines.append("")

    output_lines.append(f"**Search provider:** {SEARCH_PROVIDER}")
    output_lines.append(f"**Results found:** {len(unique_results)}")
    output_lines.append("")
    output_lines.append("---")
    output_lines.append("")
    output_lines.append(_format_results_markdown(unique_results))

    return "\n".join(output_lines)


@mcp.tool(
    name="search_web",
    annotations={
        "title": "Search the Web",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def search_web(params: SearchWebInput) -> str:
    """Run a raw web search and return results. Use this for any query
    that doesn't fit the other specialized tools.

    Args:
        params (SearchWebInput): Contains:
            - query (str): Search query
            - max_results (int): Number of results to return

    Returns:
        str: Markdown-formatted search results
    """
    results = await search(params.query, params.max_results)

    output_lines = [
        f"# Search: {params.query}",
        "",
        f"**Provider:** {SEARCH_PROVIDER}",
        f"**Results:** {len(results)}",
        "",
        "---",
        "",
        _format_results_markdown(results),
    ]

    return "\n".join(output_lines)


# --- Run ---

if __name__ == "__main__":
    mcp.run()
