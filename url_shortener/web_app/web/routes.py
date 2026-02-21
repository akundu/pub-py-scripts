"""Web interface routes implementation."""

import os
from fastapi import APIRouter, Request, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates

from lib.common.url_builder import build_short_url
from lib.common.headers import build_base_url, get_forwarded_path_prefix
from ..html_prefix import inject_forwarded_prefix_into_html

router = APIRouter()

# Setup Jinja2 templates
template_dir = os.path.join(os.path.dirname(__file__), "..", "..", "ux", "web")
templates = Jinja2Templates(directory=template_dir)


def _path_prefix_from_request(request: Request, config) -> str:
    """Path prefix from X-Forwarded-Prefix (Envoy) or config. Normalized: leading slash, no trailing."""
    prefix = get_forwarded_path_prefix(dict(request.headers))
    if prefix:
        return prefix
    p = (getattr(config, "path_prefix", "") or "").strip().strip("/")
    return "/" + p if p else ""


def _path_prefix_for_html(request: Request) -> str:
    """Path prefix only when request came through proxy (X-Forwarded-Prefix). Use for HTML rewriting so
    direct access (localhost:9200) keeps relative links and assets load from /css/, /js/."""
    return get_forwarded_path_prefix(dict(request.headers))


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def homepage(request: Request):
    """Serve the homepage. Only rewrite HTML when X-Forwarded-Prefix is set (proxy); direct access uses relative links."""
    html_file = os.path.join(template_dir, "index.html")
    config = request.app.state.config
    prefix = _path_prefix_for_html(request)
    
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            content = f.read()
        content = inject_forwarded_prefix_into_html(content, prefix)
        return HTMLResponse(content=content)
    
    return HTMLResponse(
        content="<h1>URL Shortener</h1><p>Homepage under construction</p>",
        status_code=200,
    )


@router.post("/create", response_class=HTMLResponse, include_in_schema=False)
async def create_short_url_web(
    request: Request,
    url: str = Form(...),
    custom_code: str = Form(None),
):
    """Handle form submission to create short URL."""
    service = request.app.state.service
    config = request.app.state.config
    
    # Clean up custom code (empty string to None)
    if custom_code and custom_code.strip():
        custom_code = custom_code.strip()
    else:
        custom_code = None
    
    try:
        # Create short URL
        result = await service.create_short_url(
            original_url=url,
            custom_code=custom_code,
        )
        
        # Build complete short URL (use X-Forwarded-Prefix so displayed URL includes /u_s/ when behind proxy)
        base_url = build_base_url(
            headers=dict(request.headers),
            fallback_base_url=config.base_url,
            request_scheme=request.url.scheme,
            request_host=request.headers.get("host"),
        )
        # Use prefix only when from proxy (X-Forwarded-Prefix) so short URL is correct for both direct and proxy
        path_prefix = _path_prefix_for_html(request)
        short_url = build_short_url(
            short_code=result["short_code"],
            base_url=base_url,
            path_prefix=path_prefix,
        )
        
        # Relative redirect so it works with or without proxy path (browser resolves relative to current path)
        return RedirectResponse(
            url=f"result/{result['short_code']}",
            status_code=status.HTTP_303_SEE_OTHER,
        )
        
    except ValueError as e:
        # Return error page (rewrite links only when X-Forwarded-Prefix set)
        error_file = os.path.join(template_dir, "error.html")
        path_prefix = _path_prefix_for_html(request)
        if os.path.exists(error_file):
            with open(error_file, "r", encoding="utf-8") as f:
                content = f.read()
            content = content.replace("{{error_message}}", str(e))
            content = inject_forwarded_prefix_into_html(content, path_prefix)
            return HTMLResponse(content=content, status_code=400)
        
        return HTMLResponse(
            content=f"<h1>Error</h1><p>{str(e)}</p><a href='..'>Go back</a>",
            status_code=400,
        )


@router.get("/result/{short_code}", response_class=HTMLResponse, include_in_schema=False)
async def result_page(request: Request, short_code: str):
    """Show result page with short URL."""
    service = request.app.state.service
    config = request.app.state.config
    
    # Get URL info
    info = await service.get_url_info(short_code)
    
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Short code '{short_code}' not found",
        )
    
    # Build complete short URL (use X-Forwarded-Prefix so displayed URL is e.g. https://kundu.dev/u_s/equity_percentile)
    base_url = build_base_url(
        headers=dict(request.headers),
        fallback_base_url=config.base_url,
        request_scheme=request.url.scheme,
        request_host=request.headers.get("host"),
    )
    # Use prefix only when from proxy so short URL and asset links work for both direct and proxy
    path_prefix = _path_prefix_for_html(request)
    short_url = build_short_url(
        short_code=short_code,
        base_url=base_url,
        path_prefix=path_prefix,
    )
    
    # Serve result page (rewrite asset links only when X-Forwarded-Prefix set)
    result_file = os.path.join(template_dir, "result.html")
    
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace("{{short_url}}", short_url)
        content = content.replace("{{short_code}}", short_code)
        content = content.replace("{{original_url}}", info["original_url"])
        content = inject_forwarded_prefix_into_html(content, path_prefix)
        return HTMLResponse(content=content)
    
    # Fallback (rewrite link when prefix set)
    fallback = (
        f"<h1>Success!</h1><p>Short URL: <a href='{short_url}'>{short_url}</a></p>"
        f"<p>Original: {info['original_url']}</p><a href='../..'>Create another</a>"
    )
    fallback = inject_forwarded_prefix_into_html(fallback, path_prefix)
    return HTMLResponse(content=fallback, status_code=200)


@router.get("/{short_code}", include_in_schema=False)
async def redirect_to_url(request: Request, short_code: str):
    """Redirect to the original URL."""
    service = request.app.state.service
    
    # Get original URL (this also increments access count)
    original_url = await service.get_original_url(short_code, increment_count=True)
    
    if not original_url:
        # Show 404 error page (rewrite links only when X-Forwarded-Prefix set)
        error_file = os.path.join(template_dir, "error.html")
        config = request.app.state.config
        path_prefix = _path_prefix_for_html(request)
        if os.path.exists(error_file):
            with open(error_file, "r", encoding="utf-8") as f:
                content = f.read()
            content = content.replace("{{error_message}}", f"Short code '{short_code}' not found")
            content = inject_forwarded_prefix_into_html(content, path_prefix)
            return HTMLResponse(content=content, status_code=404)
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Short code '{short_code}' not found",
        )
    
    # Perform 302 redirect (temporary redirect for tracking)
    return RedirectResponse(url=original_url, status_code=status.HTTP_302_FOUND)


@router.get("/health", include_in_schema=False)
async def health_check_web(request: Request):
    """Health check endpoint (simple version for load balancers)."""
    service = request.app.state.service
    
    health = await service.health_check()
    
    if health["overall"]:
        return {"status": "healthy"}
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy",
        )






