"""Web interface routes implementation."""

import os
from fastapi import APIRouter, Request, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates

from ...lib.common.url_builder import build_short_url
from ...lib.common.headers import build_base_url

router = APIRouter()

# Setup Jinja2 templates
template_dir = os.path.join(os.path.dirname(__file__), "..", "..", "ux", "web")
templates = Jinja2Templates(directory=template_dir)


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def homepage(request: Request):
    """Serve the homepage."""
    html_file = os.path.join(template_dir, "index.html")
    
    if os.path.exists(html_file):
        return FileResponse(html_file)
    
    # Fallback if file doesn't exist yet
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
        
        # Build complete short URL
        base_url = build_base_url(
            headers=dict(request.headers),
            fallback_base_url=config.base_url,
            request_scheme=request.url.scheme,
            request_host=request.headers.get("host"),
        )
        
        short_url = build_short_url(
            short_code=result["short_code"],
            base_url=base_url,
            path_prefix=config.path_prefix,
        )
        
        # Redirect to result page
        return RedirectResponse(
            url=f"/result/{result['short_code']}",
            status_code=status.HTTP_303_SEE_OTHER,
        )
        
    except ValueError as e:
        # Return error page
        error_file = os.path.join(template_dir, "error.html")
        if os.path.exists(error_file):
            with open(error_file, 'r') as f:
                content = f.read()
                content = content.replace("{{error_message}}", str(e))
                return HTMLResponse(content=content, status_code=400)
        
        return HTMLResponse(
            content=f"<h1>Error</h1><p>{str(e)}</p><a href='/'>Go back</a>",
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
    
    # Build complete short URL
    base_url = build_base_url(
        headers=dict(request.headers),
        fallback_base_url=config.base_url,
        request_scheme=request.url.scheme,
        request_host=request.headers.get("host"),
    )
    
    short_url = build_short_url(
        short_code=short_code,
        base_url=base_url,
        path_prefix=config.path_prefix,
    )
    
    # Serve result page
    result_file = os.path.join(template_dir, "result.html")
    
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            content = f.read()
            content = content.replace("{{short_url}}", short_url)
            content = content.replace("{{short_code}}", short_code)
            content = content.replace("{{original_url}}", info["original_url"])
            return HTMLResponse(content=content)
    
    # Fallback
    return HTMLResponse(
        content=f"<h1>Success!</h1><p>Short URL: <a href='{short_url}'>{short_url}</a></p>"
        f"<p>Original: {info['original_url']}</p><a href='/'>Create another</a>",
        status_code=200,
    )


@router.get("/{short_code}", include_in_schema=False)
async def redirect_to_url(request: Request, short_code: str):
    """Redirect to the original URL."""
    service = request.app.state.service
    
    # Get original URL (this also increments access count)
    original_url = await service.get_original_url(short_code, increment_count=True)
    
    if not original_url:
        # Show 404 error page
        error_file = os.path.join(template_dir, "error.html")
        if os.path.exists(error_file):
            with open(error_file, 'r') as f:
                content = f.read()
                content = content.replace("{{error_message}}", f"Short code '{short_code}' not found")
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





