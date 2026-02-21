"""HTML path-prefix rewriting for proxy (X-Forwarded-Prefix). No app imports to avoid circular deps."""


def inject_forwarded_prefix_into_html(html: str, path_prefix: str) -> str:
    """Rewrite relative asset/nav URLs to absolute with path_prefix so they work when Envoy strips the prefix.
    Use when X-Forwarded-Prefix is set (e.g. /u_s). Prefix should be normalized (leading slash, no trailing).
    """
    if not path_prefix:
        return html
    # Relative asset paths used in our HTML -> absolute with prefix (so browser requests /u_s/css/...)
    html = html.replace('href="../../css/', f'href="{path_prefix}/css/')
    html = html.replace('href="../css/', f'href="{path_prefix}/css/')
    html = html.replace('href="css/', f'href="{path_prefix}/css/')
    html = html.replace('src="js/', f'src="{path_prefix}/js/')
    html = html.replace('action="create"', f'action="{path_prefix}/create"')
    html = html.replace('href="../.."', f'href="{path_prefix}/"')
    html = html.replace('href=".."', f'href="{path_prefix}/"')
    return html
