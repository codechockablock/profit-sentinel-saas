"""
DORIAN INTERNET, WEB & PROGRAMMING LANGUAGES DOMAIN
====================================================

Comprehensive knowledge of:
1. Internet infrastructure and protocols
2. Web browsers and rendering engines
3. Web technologies (HTML, CSS, JS, APIs)
4. All major programming languages with deep detail
5. Web frameworks and libraries
6. Web security and standards

Author: Joseph + Claude
Date: 2026-01-25
"""

from typing import Dict, List, Set, Tuple

# =============================================================================
# INTERNET & WEB CATEGORIES
# =============================================================================

WEB_CATEGORIES = [
    # =========================================================================
    # INTERNET INFRASTRUCTURE
    # =========================================================================
    # Core internet
    (
        "internet_infrastructure",
        "system",
        "The physical and logical infrastructure of the internet",
    ),
    ("backbone", "internet_infrastructure", "High-capacity network links"),
    (
        "internet_exchange_point",
        "internet_infrastructure",
        "Where networks interconnect",
    ),
    ("ixp", "internet_exchange_point", "Internet Exchange Point"),
    ("tier_1_network", "internet_infrastructure", "Top-level internet provider"),
    ("tier_2_network", "internet_infrastructure", "Regional internet provider"),
    ("isp", "internet_infrastructure", "Internet Service Provider"),
    ("cdn", "internet_infrastructure", "Content Delivery Network"),
    ("edge_server", "server", "Server at network edge"),
    ("origin_server", "server", "Original content server"),
    ("proxy_server", "server", "Intermediary server"),
    ("reverse_proxy", "proxy_server", "Server-side proxy"),
    ("forward_proxy", "proxy_server", "Client-side proxy"),
    # DNS
    ("dns_system", "internet_infrastructure", "Domain Name System"),
    ("dns_server", "server", "Server resolving domain names"),
    ("root_dns", "dns_server", "Root DNS server"),
    ("tld_dns", "dns_server", "Top-level domain DNS server"),
    ("authoritative_dns", "dns_server", "Authoritative DNS server"),
    ("recursive_dns", "dns_server", "Recursive DNS resolver"),
    ("dns_record", "data", "A DNS record"),
    ("a_record", "dns_record", "IPv4 address record"),
    ("aaaa_record", "dns_record", "IPv6 address record"),
    ("cname_record", "dns_record", "Canonical name record"),
    ("mx_record", "dns_record", "Mail exchange record"),
    ("txt_record", "dns_record", "Text record"),
    ("ns_record", "dns_record", "Name server record"),
    ("soa_record", "dns_record", "Start of authority record"),
    # Domain names
    ("domain_name", "identifier", "Human-readable internet address"),
    ("tld", "domain_name", "Top-level domain"),
    ("gtld", "tld", "Generic TLD (.com, .org)"),
    ("cctld", "tld", "Country code TLD (.uk, .de)"),
    ("subdomain", "domain_name", "Prefix to domain name"),
    ("fqdn", "domain_name", "Fully qualified domain name"),
    # IP addressing
    ("ip_address", "identifier", "Internet Protocol address"),
    ("ipv4_address", "ip_address", "32-bit IP address"),
    ("ipv6_address", "ip_address", "128-bit IP address"),
    ("subnet", "network", "Subdivision of IP network"),
    ("cidr", "notation", "Classless Inter-Domain Routing"),
    ("nat", "technique", "Network Address Translation"),
    ("port", "identifier", "Network port number"),
    # Routing
    ("routing", "process", "Directing network packets"),
    ("bgp", "network_protocol", "Border Gateway Protocol"),
    ("ospf", "network_protocol", "Open Shortest Path First"),
    ("autonomous_system", "network", "Independently administered network"),
    ("asn", "identifier", "Autonomous System Number"),
    # =========================================================================
    # WEB PROTOCOLS
    # =========================================================================
    ("web_protocol", "network_protocol", "Protocol for the web"),
    # HTTP family
    ("http_1_0", "http", "HTTP version 1.0"),
    ("http_1_1", "http", "HTTP version 1.1"),
    ("http_2", "http", "HTTP version 2"),
    ("http_3", "http", "HTTP version 3 over QUIC"),
    ("quic", "network_protocol", "Quick UDP Internet Connections"),
    # HTTP concepts
    ("http_method", "abstract", "HTTP request method"),
    ("get_method", "http_method", "GET request"),
    ("post_method", "http_method", "POST request"),
    ("put_method", "http_method", "PUT request"),
    ("delete_method", "http_method", "DELETE request"),
    ("patch_method", "http_method", "PATCH request"),
    ("head_method", "http_method", "HEAD request"),
    ("options_method", "http_method", "OPTIONS request"),
    ("http_header", "data", "HTTP header field"),
    ("content_type", "http_header", "Content-Type header"),
    ("authorization", "http_header", "Authorization header"),
    ("cache_control", "http_header", "Cache-Control header"),
    ("cookie_header", "http_header", "Cookie header"),
    ("cors_header", "http_header", "CORS header"),
    ("http_status", "data", "HTTP status code"),
    ("status_2xx", "http_status", "Success status codes"),
    ("status_3xx", "http_status", "Redirection status codes"),
    ("status_4xx", "http_status", "Client error status codes"),
    ("status_5xx", "http_status", "Server error status codes"),
    # WebSocket
    ("websocket_protocol", "web_protocol", "Full-duplex communication"),
    ("socket_io", "websocket_protocol", "Socket.IO library"),
    # API protocols
    ("api_protocol", "web_protocol", "API communication protocol"),
    ("rest", "api_protocol", "Representational State Transfer"),
    ("graphql", "api_protocol", "Graph Query Language"),
    ("soap", "api_protocol", "Simple Object Access Protocol"),
    ("json_rpc", "api_protocol", "JSON Remote Procedure Call"),
    ("xml_rpc", "api_protocol", "XML Remote Procedure Call"),
    # Data formats
    ("data_format", "abstract", "Format for data exchange"),
    ("json", "data_format", "JavaScript Object Notation"),
    ("xml", "data_format", "Extensible Markup Language"),
    ("yaml", "data_format", "YAML Ain't Markup Language"),
    ("protobuf", "data_format", "Protocol Buffers"),
    ("msgpack", "data_format", "MessagePack"),
    ("avro", "data_format", "Apache Avro"),
    # =========================================================================
    # WEB BROWSERS
    # =========================================================================
    ("web_browser", "software", "Application for browsing the web"),
    # Major browsers
    ("chrome", "web_browser", "Google Chrome browser"),
    ("firefox", "web_browser", "Mozilla Firefox browser"),
    ("safari", "web_browser", "Apple Safari browser"),
    ("edge", "web_browser", "Microsoft Edge browser"),
    ("opera", "web_browser", "Opera browser"),
    ("brave", "web_browser", "Brave browser"),
    ("vivaldi", "web_browser", "Vivaldi browser"),
    ("arc", "web_browser", "Arc browser"),
    # Browser engines
    ("browser_engine", "software", "Core browser rendering engine"),
    ("rendering_engine", "browser_engine", "HTML/CSS renderer"),
    ("javascript_engine", "browser_engine", "JavaScript executor"),
    # Rendering engines
    ("blink", "rendering_engine", "Chrome's rendering engine"),
    ("webkit", "rendering_engine", "Safari's rendering engine"),
    ("gecko", "rendering_engine", "Firefox's rendering engine"),
    ("trident", "rendering_engine", "Legacy IE engine"),
    ("edgehtml", "rendering_engine", "Legacy Edge engine"),
    # JavaScript engines
    ("v8", "javascript_engine", "Chrome's JS engine"),
    ("spidermonkey", "javascript_engine", "Firefox's JS engine"),
    ("javascriptcore", "javascript_engine", "Safari's JS engine"),
    ("chakra", "javascript_engine", "Legacy Edge JS engine"),
    # Browser components
    ("browser_component", "software", "Component of a browser"),
    ("dom_parser", "browser_component", "Parses HTML to DOM"),
    ("css_parser", "browser_component", "Parses CSS"),
    ("layout_engine", "browser_component", "Calculates element positions"),
    ("paint_engine", "browser_component", "Renders pixels"),
    ("compositor", "browser_component", "Composites layers"),
    ("network_stack", "browser_component", "Handles network requests"),
    ("cache_storage", "browser_component", "Browser cache"),
    ("cookie_storage", "browser_component", "Cookie storage"),
    ("local_storage", "browser_component", "localStorage API"),
    ("session_storage", "browser_component", "sessionStorage API"),
    ("indexed_db", "browser_component", "IndexedDB storage"),
    # Browser APIs
    ("browser_api", "api", "Browser-provided API"),
    ("dom_api", "browser_api", "Document Object Model API"),
    ("fetch_api", "browser_api", "Fetch API for requests"),
    ("canvas_api", "browser_api", "2D drawing API"),
    ("webgl", "browser_api", "3D graphics API"),
    ("webgpu", "browser_api", "Modern GPU API"),
    ("web_audio", "browser_api", "Audio processing API"),
    ("web_rtc", "browser_api", "Real-time communication"),
    ("service_worker", "browser_api", "Background worker API"),
    ("web_worker", "browser_api", "Background thread API"),
    ("push_api", "browser_api", "Push notification API"),
    ("geolocation_api", "browser_api", "Location API"),
    ("file_api", "browser_api", "File system access"),
    ("clipboard_api", "browser_api", "Clipboard access"),
    ("payment_api", "browser_api", "Payment Request API"),
    ("web_share", "browser_api", "Share API"),
    ("web_bluetooth", "browser_api", "Bluetooth API"),
    ("web_usb", "browser_api", "USB device API"),
    # =========================================================================
    # WEB TECHNOLOGIES - HTML
    # =========================================================================
    ("markup_language", "programming_language", "Language using tags"),
    ("html", "markup_language", "HyperText Markup Language"),
    ("html5", "html", "HTML version 5"),
    ("xhtml", "html", "XML-based HTML"),
    # HTML elements
    ("html_element", "abstract", "An HTML element"),
    ("semantic_element", "html_element", "Meaningful HTML element"),
    ("form_element", "html_element", "Form-related element"),
    ("media_element", "html_element", "Media playback element"),
    ("interactive_element", "html_element", "Interactive element"),
    # Document structure
    ("doctype", "html_element", "Document type declaration"),
    ("html_tag", "html_element", "Root HTML element"),
    ("head_tag", "html_element", "Document head"),
    ("body_tag", "html_element", "Document body"),
    ("meta_tag", "html_element", "Metadata element"),
    ("link_tag", "html_element", "External resource link"),
    ("script_tag", "html_element", "JavaScript element"),
    ("style_tag", "html_element", "CSS element"),
    # Semantic elements
    ("header_tag", "semantic_element", "Header section"),
    ("nav_tag", "semantic_element", "Navigation section"),
    ("main_tag", "semantic_element", "Main content"),
    ("article_tag", "semantic_element", "Article content"),
    ("section_tag", "semantic_element", "Generic section"),
    ("aside_tag", "semantic_element", "Sidebar content"),
    ("footer_tag", "semantic_element", "Footer section"),
    # =========================================================================
    # WEB TECHNOLOGIES - CSS
    # =========================================================================
    ("stylesheet_language", "programming_language", "Language for styling"),
    ("css", "stylesheet_language", "Cascading Style Sheets"),
    ("css3", "css", "CSS version 3"),
    # CSS concepts
    ("css_concept", "abstract", "A CSS concept"),
    ("selector", "css_concept", "CSS selector"),
    ("property", "css_concept", "CSS property"),
    ("value", "css_concept", "CSS value"),
    ("specificity", "css_concept", "Selector priority"),
    ("cascade", "css_concept", "Style inheritance"),
    ("box_model", "css_concept", "Element box model"),
    # Layout systems
    ("css_layout", "css_concept", "CSS layout system"),
    ("flexbox", "css_layout", "Flexible box layout"),
    ("css_grid", "css_layout", "Grid layout"),
    ("float_layout", "css_layout", "Float-based layout"),
    ("position_layout", "css_layout", "Position-based layout"),
    # CSS features
    ("css_feature", "css_concept", "CSS feature"),
    ("media_query", "css_feature", "Responsive breakpoints"),
    ("css_variable", "css_feature", "Custom properties"),
    ("css_animation", "css_feature", "Keyframe animation"),
    ("css_transition", "css_feature", "Property transition"),
    ("css_transform", "css_feature", "Element transformation"),
    ("css_filter", "css_feature", "Visual filter effects"),
    # CSS preprocessors
    ("css_preprocessor", "software", "CSS preprocessor"),
    ("sass", "css_preprocessor", "Syntactically Awesome Stylesheets"),
    ("scss", "sass", "Sassy CSS"),
    ("less", "css_preprocessor", "Leaner CSS"),
    ("stylus", "css_preprocessor", "Stylus preprocessor"),
    ("postcss", "css_preprocessor", "PostCSS processor"),
    # CSS frameworks
    ("css_framework", "software", "CSS framework"),
    ("tailwind", "css_framework", "Tailwind CSS"),
    ("bootstrap", "css_framework", "Bootstrap framework"),
    ("bulma", "css_framework", "Bulma framework"),
    ("foundation", "css_framework", "Foundation framework"),
    ("materialize", "css_framework", "Material Design CSS"),
    # =========================================================================
    # WEB TECHNOLOGIES - JAVASCRIPT ECOSYSTEM
    # =========================================================================
    # JavaScript core
    ("ecmascript", "programming_language", "ECMAScript specification"),
    ("es5", "ecmascript", "ECMAScript 5"),
    ("es6", "ecmascript", "ECMAScript 2015"),
    ("es2015", "es6", "ECMAScript 2015"),
    ("es2016", "ecmascript", "ECMAScript 2016"),
    ("es2017", "ecmascript", "ECMAScript 2017"),
    ("es2018", "ecmascript", "ECMAScript 2018"),
    ("es2019", "ecmascript", "ECMAScript 2019"),
    ("es2020", "ecmascript", "ECMAScript 2020"),
    ("es2021", "ecmascript", "ECMAScript 2021"),
    ("es2022", "ecmascript", "ECMAScript 2022"),
    ("es2023", "ecmascript", "ECMAScript 2023"),
    # JS concepts
    ("js_concept", "abstract", "JavaScript concept"),
    ("prototype", "js_concept", "Prototype inheritance"),
    ("closure", "js_concept", "Function closure"),
    ("promise", "js_concept", "Async promise"),
    ("async_await", "js_concept", "Async/await syntax"),
    ("event_loop", "js_concept", "JavaScript event loop"),
    ("callback", "js_concept", "Callback function"),
    ("hoisting", "js_concept", "Variable hoisting"),
    ("scope", "js_concept", "Variable scope"),
    ("this_keyword", "js_concept", "this binding"),
    # Package managers
    ("package_manager", "software", "Dependency manager"),
    ("npm", "package_manager", "Node Package Manager"),
    ("yarn", "package_manager", "Yarn package manager"),
    ("pnpm", "package_manager", "Performant npm"),
    ("bun_pm", "package_manager", "Bun package manager"),
    # Build tools
    ("build_tool", "software", "Build and bundle tool"),
    ("webpack", "build_tool", "Webpack bundler"),
    ("vite", "build_tool", "Vite build tool"),
    ("rollup", "build_tool", "Rollup bundler"),
    ("parcel", "build_tool", "Parcel bundler"),
    ("esbuild", "build_tool", "esbuild bundler"),
    ("swc", "build_tool", "SWC compiler"),
    ("babel", "build_tool", "Babel transpiler"),
    ("terser", "build_tool", "Terser minifier"),
    # JS runtimes
    ("js_runtime", "software", "JavaScript runtime"),
    ("nodejs", "js_runtime", "Node.js runtime"),
    ("deno", "js_runtime", "Deno runtime"),
    ("bun", "js_runtime", "Bun runtime"),
    # Frontend frameworks
    ("frontend_framework", "software", "Frontend JavaScript framework"),
    ("react", "frontend_framework", "React library"),
    ("vue", "frontend_framework", "Vue.js framework"),
    ("angular", "frontend_framework", "Angular framework"),
    ("svelte", "frontend_framework", "Svelte framework"),
    ("solid", "frontend_framework", "SolidJS framework"),
    ("preact", "frontend_framework", "Preact library"),
    ("qwik", "frontend_framework", "Qwik framework"),
    ("htmx", "frontend_framework", "htmx library"),
    # React ecosystem
    ("react_ecosystem", "software", "React-related library"),
    ("next_js", "react_ecosystem", "Next.js framework"),
    ("remix", "react_ecosystem", "Remix framework"),
    ("gatsby", "react_ecosystem", "Gatsby framework"),
    ("react_router", "react_ecosystem", "React Router"),
    ("redux", "react_ecosystem", "Redux state management"),
    ("zustand", "react_ecosystem", "Zustand state management"),
    ("react_query", "react_ecosystem", "React Query"),
    ("swr", "react_ecosystem", "SWR data fetching"),
    # Vue ecosystem
    ("vue_ecosystem", "software", "Vue-related library"),
    ("nuxt", "vue_ecosystem", "Nuxt.js framework"),
    ("vuex", "vue_ecosystem", "Vuex state management"),
    ("pinia", "vue_ecosystem", "Pinia state management"),
    ("vue_router", "vue_ecosystem", "Vue Router"),
    # Testing
    ("testing_framework", "software", "Testing framework"),
    ("jest", "testing_framework", "Jest testing"),
    ("vitest", "testing_framework", "Vitest testing"),
    ("mocha", "testing_framework", "Mocha testing"),
    ("cypress", "testing_framework", "Cypress E2E testing"),
    ("playwright", "testing_framework", "Playwright testing"),
    ("puppeteer", "testing_framework", "Puppeteer browser automation"),
    # =========================================================================
    # BACKEND FRAMEWORKS
    # =========================================================================
    ("backend_framework", "software", "Server-side framework"),
    # Node.js frameworks
    ("express", "backend_framework", "Express.js framework"),
    ("fastify", "backend_framework", "Fastify framework"),
    ("nest_js", "backend_framework", "NestJS framework"),
    ("koa", "backend_framework", "Koa framework"),
    ("hapi", "backend_framework", "Hapi framework"),
    # Python frameworks
    ("django", "backend_framework", "Django framework"),
    ("flask", "backend_framework", "Flask framework"),
    ("fastapi", "backend_framework", "FastAPI framework"),
    ("tornado", "backend_framework", "Tornado framework"),
    ("starlette", "backend_framework", "Starlette framework"),
    # Other backend
    ("rails", "backend_framework", "Ruby on Rails"),
    ("laravel", "backend_framework", "Laravel PHP framework"),
    ("spring", "backend_framework", "Spring Java framework"),
    ("spring_boot", "spring", "Spring Boot"),
    ("asp_net", "backend_framework", "ASP.NET framework"),
    ("gin", "backend_framework", "Gin Go framework"),
    ("fiber", "backend_framework", "Fiber Go framework"),
    ("actix", "backend_framework", "Actix Rust framework"),
    ("axum", "backend_framework", "Axum Rust framework"),
    # =========================================================================
    # WEB SECURITY
    # =========================================================================
    ("web_security", "security", "Web application security"),
    # Vulnerabilities
    ("web_vulnerability", "cyber_attack", "Web security vulnerability"),
    ("xss", "web_vulnerability", "Cross-Site Scripting"),
    ("stored_xss", "xss", "Stored XSS"),
    ("reflected_xss", "xss", "Reflected XSS"),
    ("dom_xss", "xss", "DOM-based XSS"),
    ("csrf", "web_vulnerability", "Cross-Site Request Forgery"),
    ("sql_injection", "web_vulnerability", "SQL Injection"),
    ("xxe", "web_vulnerability", "XML External Entity"),
    ("ssrf", "web_vulnerability", "Server-Side Request Forgery"),
    ("idor", "web_vulnerability", "Insecure Direct Object Reference"),
    ("rce", "web_vulnerability", "Remote Code Execution"),
    ("lfi", "web_vulnerability", "Local File Inclusion"),
    ("rfi", "web_vulnerability", "Remote File Inclusion"),
    ("clickjacking", "web_vulnerability", "UI redress attack"),
    # Security headers
    ("security_header", "http_header", "Security HTTP header"),
    ("csp", "security_header", "Content Security Policy"),
    ("hsts", "security_header", "HTTP Strict Transport Security"),
    ("x_frame_options", "security_header", "X-Frame-Options"),
    ("x_content_type", "security_header", "X-Content-Type-Options"),
    # Authentication
    ("web_auth", "authentication", "Web authentication"),
    ("session_auth", "web_auth", "Session-based auth"),
    ("token_auth", "web_auth", "Token-based auth"),
    ("oauth_2", "web_auth", "OAuth 2.0"),
    ("oidc", "web_auth", "OpenID Connect"),
    ("saml", "web_auth", "SAML authentication"),
    ("passkey", "web_auth", "Passkey authentication"),
    ("webauthn", "web_auth", "Web Authentication API"),
    # =========================================================================
    # WEB STANDARDS
    # =========================================================================
    ("web_standard", "abstract", "Web standard"),
    ("w3c_standard", "web_standard", "W3C standard"),
    ("whatwg_standard", "web_standard", "WHATWG standard"),
    ("ecma_standard", "web_standard", "ECMA standard"),
    ("ietf_standard", "web_standard", "IETF standard"),
    ("rfc", "ietf_standard", "Request for Comments"),
    # =========================================================================
    # CLOUD & HOSTING
    # =========================================================================
    ("cloud_provider", "internet_infrastructure", "Cloud service provider"),
    ("aws", "cloud_provider", "Amazon Web Services"),
    ("gcp", "cloud_provider", "Google Cloud Platform"),
    ("azure", "cloud_provider", "Microsoft Azure"),
    ("cloudflare", "cloud_provider", "Cloudflare"),
    ("vercel", "cloud_provider", "Vercel platform"),
    ("netlify", "cloud_provider", "Netlify platform"),
    ("heroku", "cloud_provider", "Heroku platform"),
    ("digitalocean", "cloud_provider", "DigitalOcean"),
    ("linode", "cloud_provider", "Linode"),
    # Cloud services
    ("cloud_service", "software", "Cloud-provided service"),
    ("s3", "cloud_service", "AWS S3 storage"),
    ("ec2", "cloud_service", "AWS EC2 compute"),
    ("lambda", "cloud_service", "AWS Lambda serverless"),
    ("cloudfront", "cloud_service", "AWS CloudFront CDN"),
    ("cloud_functions", "cloud_service", "Google Cloud Functions"),
    ("cloud_run", "cloud_service", "Google Cloud Run"),
]


# =============================================================================
# PROGRAMMING LANGUAGES - COMPREHENSIVE
# =============================================================================

LANG_CATEGORIES = [
    # =========================================================================
    # LANGUAGE CLASSIFICATION
    # =========================================================================
    ("language_classification", "abstract", "Way to classify languages"),
    # By paradigm (extending existing)
    ("multi_paradigm", "language_paradigm", "Supports multiple paradigms"),
    ("event_driven", "language_paradigm", "Event-driven programming"),
    ("reactive", "language_paradigm", "Reactive programming"),
    ("generic", "language_paradigm", "Generic programming"),
    ("metaprogramming", "language_paradigm", "Code that manipulates code"),
    ("reflective", "language_paradigm", "Self-inspection at runtime"),
    ("aspect_oriented", "language_paradigm", "Cross-cutting concerns"),
    # By typing
    ("type_system", "abstract", "Language type system"),
    ("static_typing", "type_system", "Types checked at compile time"),
    ("dynamic_typing", "type_system", "Types checked at runtime"),
    ("strong_typing", "type_system", "Strict type enforcement"),
    ("weak_typing", "type_system", "Loose type enforcement"),
    ("duck_typing", "type_system", "Types by behavior"),
    ("gradual_typing", "type_system", "Optional type annotations"),
    ("dependent_typing", "type_system", "Types depending on values"),
    ("structural_typing", "type_system", "Types by structure"),
    ("nominal_typing", "type_system", "Types by name"),
    # By memory management
    ("memory_management", "abstract", "How language manages memory"),
    ("manual_memory", "memory_management", "Manual memory management"),
    ("garbage_collected", "memory_management", "Automatic garbage collection"),
    ("reference_counted", "memory_management", "Reference counting"),
    ("ownership_based", "memory_management", "Ownership and borrowing"),
    ("arena_allocation", "memory_management", "Arena/region allocation"),
    # By execution
    ("execution_model", "abstract", "How code is executed"),
    ("ahead_of_time", "execution_model", "AOT compilation"),
    ("just_in_time", "execution_model", "JIT compilation"),
    ("interpreted_execution", "execution_model", "Interpretation"),
    ("bytecode_execution", "execution_model", "Bytecode VM"),
    ("transpiled", "execution_model", "Compiled to another language"),
    # =========================================================================
    # SYSTEMS LANGUAGES
    # =========================================================================
    ("systems_language", "programming_language", "Language for systems programming"),
    # C family
    ("c_language", "systems_language", "C programming language"),
    ("c89", "c_language", "C89/ANSI C standard"),
    ("c99", "c_language", "C99 standard"),
    ("c11", "c_language", "C11 standard"),
    ("c17", "c_language", "C17 standard"),
    ("c23", "c_language", "C23 standard"),
    ("cpp_language", "systems_language", "C++ programming language"),
    ("cpp98", "cpp_language", "C++98 standard"),
    ("cpp03", "cpp_language", "C++03 standard"),
    ("cpp11", "cpp_language", "C++11 standard"),
    ("cpp14", "cpp_language", "C++14 standard"),
    ("cpp17", "cpp_language", "C++17 standard"),
    ("cpp20", "cpp_language", "C++20 standard"),
    ("cpp23", "cpp_language", "C++23 standard"),
    # Modern systems
    ("rust_language", "systems_language", "Rust programming language"),
    ("zig", "systems_language", "Zig programming language"),
    ("nim", "systems_language", "Nim programming language"),
    ("d_language", "systems_language", "D programming language"),
    ("carbon", "systems_language", "Carbon language (Google)"),
    ("odin", "systems_language", "Odin programming language"),
    ("v_language", "systems_language", "V programming language"),
    # =========================================================================
    # JVM LANGUAGES
    # =========================================================================
    ("jvm_language", "programming_language", "Language running on JVM"),
    ("java_language", "jvm_language", "Java programming language"),
    ("java_8", "java_language", "Java 8"),
    ("java_11", "java_language", "Java 11 LTS"),
    ("java_17", "java_language", "Java 17 LTS"),
    ("java_21", "java_language", "Java 21 LTS"),
    ("kotlin_language", "jvm_language", "Kotlin programming language"),
    ("scala_language", "jvm_language", "Scala programming language"),
    ("scala_2", "scala_language", "Scala 2.x"),
    ("scala_3", "scala_language", "Scala 3"),
    ("groovy", "jvm_language", "Groovy programming language"),
    ("clojure", "jvm_language", "Clojure programming language"),
    # =========================================================================
    # .NET LANGUAGES
    # =========================================================================
    ("dotnet_language", "programming_language", "Language running on .NET"),
    ("csharp_language", "dotnet_language", "C# programming language"),
    ("csharp_8", "csharp_language", "C# 8"),
    ("csharp_9", "csharp_language", "C# 9"),
    ("csharp_10", "csharp_language", "C# 10"),
    ("csharp_11", "csharp_language", "C# 11"),
    ("csharp_12", "csharp_language", "C# 12"),
    ("fsharp", "dotnet_language", "F# programming language"),
    ("vb_net", "dotnet_language", "Visual Basic .NET"),
    # =========================================================================
    # SCRIPTING LANGUAGES
    # =========================================================================
    ("scripting_language", "programming_language", "High-level scripting language"),
    # Python
    ("python_language", "scripting_language", "Python programming language"),
    ("python_2", "python_language", "Python 2.x"),
    ("python_3", "python_language", "Python 3.x"),
    ("python_3_8", "python_3", "Python 3.8"),
    ("python_3_9", "python_3", "Python 3.9"),
    ("python_3_10", "python_3", "Python 3.10"),
    ("python_3_11", "python_3", "Python 3.11"),
    ("python_3_12", "python_3", "Python 3.12"),
    ("cpython", "python_language", "Reference Python implementation"),
    ("pypy", "python_language", "JIT Python implementation"),
    ("cython", "python_language", "C-extension Python"),
    ("mypyc", "python_language", "Type-compiled Python"),
    # Ruby
    ("ruby_language", "scripting_language", "Ruby programming language"),
    ("ruby_2", "ruby_language", "Ruby 2.x"),
    ("ruby_3", "ruby_language", "Ruby 3.x"),
    ("jruby", "ruby_language", "JVM Ruby"),
    ("truffleruby", "ruby_language", "GraalVM Ruby"),
    # PHP
    ("php_language", "scripting_language", "PHP programming language"),
    ("php_7", "php_language", "PHP 7.x"),
    ("php_8", "php_language", "PHP 8.x"),
    # Perl
    ("perl_language", "scripting_language", "Perl programming language"),
    ("perl_5", "perl_language", "Perl 5"),
    ("raku", "scripting_language", "Raku (Perl 6)"),
    # Shell
    ("shell_language", "scripting_language", "Shell scripting"),
    ("bash_language", "shell_language", "Bash shell"),
    ("zsh_language", "shell_language", "Zsh shell"),
    ("fish_language", "shell_language", "Fish shell"),
    ("powershell", "shell_language", "PowerShell"),
    # Other scripting
    ("lua", "scripting_language", "Lua programming language"),
    ("luajit", "lua", "LuaJIT implementation"),
    ("tcl", "scripting_language", "Tcl programming language"),
    ("awk", "scripting_language", "AWK programming language"),
    ("sed", "scripting_language", "Sed stream editor"),
    # =========================================================================
    # FUNCTIONAL LANGUAGES
    # =========================================================================
    ("functional_language", "programming_language", "Functional programming language"),
    ("haskell_language", "functional_language", "Haskell programming language"),
    ("ghc", "haskell_language", "Glasgow Haskell Compiler"),
    ("ocaml", "functional_language", "OCaml programming language"),
    ("reasonml", "ocaml", "ReasonML syntax"),
    ("rescript", "ocaml", "ReScript language"),
    ("erlang", "functional_language", "Erlang programming language"),
    ("elixir", "functional_language", "Elixir programming language"),
    ("gleam", "functional_language", "Gleam programming language"),
    ("ml", "functional_language", "ML family"),
    ("sml", "ml", "Standard ML"),
    ("f_star", "ml", "F* proof language"),
    ("lisp_family", "functional_language", "Lisp family of languages"),
    ("common_lisp", "lisp_family", "Common Lisp"),
    ("scheme", "lisp_family", "Scheme programming language"),
    ("racket", "scheme", "Racket programming language"),
    ("emacs_lisp", "lisp_family", "Emacs Lisp"),
    # =========================================================================
    # LOGIC & DECLARATIVE LANGUAGES
    # =========================================================================
    ("logic_language", "programming_language", "Logic programming language"),
    ("prolog_language", "logic_language", "Prolog programming language"),
    ("swi_prolog", "prolog_language", "SWI-Prolog"),
    ("mercury", "logic_language", "Mercury programming language"),
    ("datalog", "logic_language", "Datalog query language"),
    # =========================================================================
    # DATA & QUERY LANGUAGES
    # =========================================================================
    ("query_language", "programming_language", "Database query language"),
    ("sql_language", "query_language", "SQL programming language"),
    ("t_sql", "sql_language", "Transact-SQL (Microsoft)"),
    ("pl_sql", "sql_language", "PL/SQL (Oracle)"),
    ("pl_pgsql", "sql_language", "PL/pgSQL (PostgreSQL)"),
    ("nosql_query", "query_language", "NoSQL query language"),
    ("mongodb_query", "nosql_query", "MongoDB query language"),
    ("cql", "nosql_query", "Cassandra Query Language"),
    ("cypher", "nosql_query", "Neo4j Cypher"),
    ("sparql", "query_language", "RDF query language"),
    # =========================================================================
    # DOMAIN-SPECIFIC LANGUAGES
    # =========================================================================
    ("dsl", "programming_language", "Domain-specific language"),
    # Config languages
    ("config_language", "dsl", "Configuration language"),
    ("json_language", "config_language", "JSON configuration"),
    ("yaml_language", "config_language", "YAML configuration"),
    ("toml", "config_language", "TOML configuration"),
    ("ini", "config_language", "INI configuration"),
    ("hcl", "config_language", "HashiCorp Configuration Language"),
    ("terraform", "hcl", "Terraform language"),
    ("jsonnet", "config_language", "Jsonnet templating"),
    ("dhall", "config_language", "Dhall configuration"),
    ("cue", "config_language", "CUE configuration"),
    # Build languages
    ("build_language", "dsl", "Build specification language"),
    ("makefile", "build_language", "Make build language"),
    ("cmake", "build_language", "CMake build language"),
    ("gradle_dsl", "build_language", "Gradle build DSL"),
    ("bazel_starlark", "build_language", "Bazel Starlark"),
    ("meson", "build_language", "Meson build language"),
    ("ninja", "build_language", "Ninja build"),
    # Data languages
    ("data_language", "dsl", "Data definition language"),
    ("regex", "data_language", "Regular expressions"),
    ("xpath", "data_language", "XPath query language"),
    ("xslt", "data_language", "XSL Transformations"),
    ("jq", "data_language", "JQ JSON processor"),
    # Document languages
    ("document_language", "dsl", "Document markup language"),
    ("markdown", "document_language", "Markdown markup"),
    ("restructuredtext", "document_language", "reStructuredText"),
    ("asciidoc", "document_language", "AsciiDoc markup"),
    ("latex", "document_language", "LaTeX typesetting"),
    ("typst", "document_language", "Typst typesetting"),
    # =========================================================================
    # GPU & PARALLEL LANGUAGES
    # =========================================================================
    ("gpu_language", "programming_language", "GPU programming language"),
    ("cuda_language", "gpu_language", "CUDA language"),
    ("opencl_language", "gpu_language", "OpenCL language"),
    ("hlsl", "gpu_language", "High-Level Shading Language"),
    ("glsl", "gpu_language", "OpenGL Shading Language"),
    ("wgsl", "gpu_language", "WebGPU Shading Language"),
    ("metal_sl", "gpu_language", "Metal Shading Language"),
    ("spirv", "gpu_language", "SPIR-V intermediate"),
    # =========================================================================
    # EMERGING & EXPERIMENTAL
    # =========================================================================
    ("emerging_language", "programming_language", "Emerging programming language"),
    ("mojo", "emerging_language", "Mojo programming language"),
    ("bend", "emerging_language", "Bend parallel language"),
    ("vale", "emerging_language", "Vale programming language"),
    ("unison", "emerging_language", "Unison content-addressed language"),
    ("dark", "emerging_language", "Dark programming language"),
    ("ballerina", "emerging_language", "Ballerina integration language"),
    ("bosque", "emerging_language", "Bosque programming language"),
    # =========================================================================
    # LANGUAGE TOOLING
    # =========================================================================
    ("language_tool", "software", "Language development tool"),
    # Compilers
    ("compiler", "language_tool", "Source to binary compiler"),
    ("gcc", "compiler", "GNU Compiler Collection"),
    ("clang", "compiler", "LLVM C/C++ compiler"),
    ("llvm", "compiler", "LLVM compiler infrastructure"),
    ("rustc", "compiler", "Rust compiler"),
    ("javac", "compiler", "Java compiler"),
    ("tsc", "compiler", "TypeScript compiler"),
    ("ghc_compiler", "compiler", "GHC Haskell compiler"),
    # Interpreters
    ("interpreter", "language_tool", "Source code interpreter"),
    ("cpython_interpreter", "interpreter", "CPython interpreter"),
    ("ruby_interpreter", "interpreter", "Ruby interpreter"),
    ("node_interpreter", "interpreter", "Node.js interpreter"),
    # Language servers
    ("language_server", "language_tool", "Language Server Protocol"),
    ("lsp", "language_server", "Language Server Protocol"),
    ("pyright", "language_server", "Python language server"),
    ("rust_analyzer", "language_server", "Rust language server"),
    ("tsserver", "language_server", "TypeScript server"),
    # Linters and formatters
    ("linter", "language_tool", "Code linter"),
    ("eslint", "linter", "JavaScript linter"),
    ("pylint", "linter", "Python linter"),
    ("ruff", "linter", "Fast Python linter"),
    ("clippy", "linter", "Rust linter"),
    ("formatter", "language_tool", "Code formatter"),
    ("prettier", "formatter", "JavaScript formatter"),
    ("black", "formatter", "Python formatter"),
    ("rustfmt", "formatter", "Rust formatter"),
    ("gofmt", "formatter", "Go formatter"),
]


# =============================================================================
# FACTS
# =============================================================================

WEB_FACTS = [
    # =========================================================================
    # INTERNET INFRASTRUCTURE
    # =========================================================================
    # DNS
    ("dns", "resolves", "domain_name"),
    ("dns", "returns", "ip_address"),
    ("a_record", "maps_to", "ipv4_address"),
    ("aaaa_record", "maps_to", "ipv6_address"),
    ("cname_record", "creates", "alias"),
    # IP
    ("ipv4_address", "has_bits", "32"),
    ("ipv6_address", "has_bits", "128"),
    ("ipv6", "successor_of", "ipv4"),
    ("nat", "translates", "ip_address"),
    # CDN
    ("cdn", "caches", "content"),
    ("cdn", "reduces", "latency"),
    ("cloudflare", "is_a", "cdn"),
    ("cloudfront", "is_a", "cdn"),
    # =========================================================================
    # HTTP
    # =========================================================================
    ("http", "is_a", "stateless_protocol"),
    ("http_2", "uses", "multiplexing"),
    ("http_2", "uses", "header_compression"),
    ("http_3", "uses", "quic"),
    ("quic", "uses", "udp"),
    ("https", "uses", "tls"),
    # Methods
    ("get_method", "is_a", "safe_method"),
    ("get_method", "is_a", "idempotent"),
    ("post_method", "is_not", "idempotent"),
    ("put_method", "is_a", "idempotent"),
    ("delete_method", "is_a", "idempotent"),
    # Status codes
    ("status_200", "means", "ok"),
    ("status_201", "means", "created"),
    ("status_301", "means", "moved_permanently"),
    ("status_302", "means", "found"),
    ("status_400", "means", "bad_request"),
    ("status_401", "means", "unauthorized"),
    ("status_403", "means", "forbidden"),
    ("status_404", "means", "not_found"),
    ("status_500", "means", "internal_server_error"),
    # =========================================================================
    # BROWSERS
    # =========================================================================
    # Engine relationships
    ("chrome", "uses", "blink"),
    ("chrome", "uses", "v8"),
    ("firefox", "uses", "gecko"),
    ("firefox", "uses", "spidermonkey"),
    ("safari", "uses", "webkit"),
    ("safari", "uses", "javascriptcore"),
    ("edge", "uses", "blink"),
    ("edge", "uses", "v8"),
    ("brave", "uses", "blink"),
    ("opera", "uses", "blink"),
    # Engine origins
    ("blink", "forked_from", "webkit"),
    ("webkit", "based_on", "khtml"),
    ("v8", "created_by", "google"),
    ("spidermonkey", "created_by", "mozilla"),
    # Browser APIs
    ("fetch_api", "replaced", "xmlhttprequest"),
    ("webgl", "based_on", "opengl"),
    ("webgpu", "successor_of", "webgl"),
    ("service_worker", "enables", "offline_functionality"),
    ("web_rtc", "enables", "peer_to_peer"),
    # =========================================================================
    # WEB TECHNOLOGIES
    # =========================================================================
    # HTML
    ("html", "defines", "document_structure"),
    ("html5", "added", "semantic_elements"),
    ("html5", "added", "canvas_api"),
    ("html5", "added", "web_audio"),
    # CSS
    ("css", "defines", "presentation"),
    ("flexbox", "is_a", "one_dimensional_layout"),
    ("css_grid", "is_a", "two_dimensional_layout"),
    ("media_query", "enables", "responsive_design"),
    ("css_variable", "enables", "dynamic_styling"),
    # JavaScript
    ("javascript", "is_a", "ecmascript"),
    ("es6", "added", "arrow_functions"),
    ("es6", "added", "class_syntax"),
    ("es6", "added", "promise"),
    ("es6", "added", "modules"),
    ("async_await", "added_in", "es2017"),
    # =========================================================================
    # FRAMEWORKS
    # =========================================================================
    # React
    ("react", "uses", "virtual_dom"),
    ("react", "uses", "jsx"),
    ("react", "created_by", "facebook"),
    ("next_js", "built_on", "react"),
    ("gatsby", "built_on", "react"),
    # Vue
    ("vue", "uses", "virtual_dom"),
    ("vue", "created_by", "evan_you"),
    ("nuxt", "built_on", "vue"),
    # Angular
    ("angular", "uses", "typescript"),
    ("angular", "created_by", "google"),
    ("angular", "uses", "dependency_injection"),
    # Svelte
    ("svelte", "compiles_to", "vanilla_javascript"),
    ("svelte", "has_no", "virtual_dom"),
    # Build tools
    ("webpack", "performs", "bundling"),
    ("vite", "uses", "esbuild"),
    ("vite", "uses", "rollup"),
    ("babel", "performs", "transpilation"),
    ("esbuild", "written_in", "go"),
    ("swc", "written_in", "rust"),
    # =========================================================================
    # BACKEND
    # =========================================================================
    # Node.js
    ("nodejs", "uses", "v8"),
    ("nodejs", "uses", "event_loop"),
    ("express", "runs_on", "nodejs"),
    ("nest_js", "uses", "typescript"),
    # Python
    ("django", "is_a", "full_stack_framework"),
    ("flask", "is_a", "micro_framework"),
    ("fastapi", "uses", "async"),
    ("fastapi", "uses", "type_hints"),
    # =========================================================================
    # SECURITY
    # =========================================================================
    ("xss", "injects", "malicious_script"),
    ("csrf", "exploits", "authenticated_session"),
    ("sql_injection", "exploits", "unsanitized_input"),
    ("csp", "prevents", "xss"),
    ("hsts", "enforces", "https"),
    ("cors", "controls", "cross_origin_requests"),
    # Auth
    ("oauth_2", "provides", "delegated_authorization"),
    ("oidc", "extends", "oauth_2"),
    ("jwt", "is_a", "self_contained_token"),
    ("webauthn", "uses", "public_key_cryptography"),
]

LANG_FACTS = [
    # =========================================================================
    # LANGUAGE PROPERTIES
    # =========================================================================
    # C
    ("c_language", "has_property", "manual_memory"),
    ("c_language", "has_property", "static_typing"),
    ("c_language", "compiles_to", "machine_code"),
    ("c_language", "created_by", "dennis_ritchie"),
    ("c_language", "created_in", "1972"),
    # C++
    ("cpp_language", "superset_of", "c_language"),
    ("cpp_language", "has_property", "object_oriented"),
    ("cpp_language", "has_property", "generic"),
    ("cpp_language", "created_by", "bjarne_stroustrup"),
    # Rust
    ("rust_language", "has_property", "ownership_based"),
    ("rust_language", "has_property", "static_typing"),
    ("rust_language", "has_no", "garbage_collector"),
    ("rust_language", "guarantees", "memory_safety"),
    ("rust_language", "created_by", "mozilla"),
    ("rust_language", "compiled_by", "rustc"),
    # Go
    ("go", "has_property", "garbage_collected"),
    ("go", "has_property", "static_typing"),
    ("go", "has_builtin", "concurrency"),
    ("go", "created_by", "google"),
    ("go", "uses", "goroutines"),
    # Java
    ("java_language", "has_property", "garbage_collected"),
    ("java_language", "runs_on", "jvm"),
    ("java_language", "has_property", "static_typing"),
    ("java_language", "created_by", "sun_microsystems"),
    ("jvm", "executes", "bytecode"),
    # Kotlin
    ("kotlin_language", "runs_on", "jvm"),
    ("kotlin_language", "interops_with", "java_language"),
    ("kotlin_language", "created_by", "jetbrains"),
    ("kotlin_language", "has_property", "null_safety"),
    # Python
    ("python_language", "has_property", "dynamic_typing"),
    ("python_language", "has_property", "garbage_collected"),
    ("python_language", "created_by", "guido_van_rossum"),
    ("python_language", "uses", "indentation_syntax"),
    ("cpython", "is_a", "reference_implementation"),
    ("pypy", "uses", "just_in_time"),
    # JavaScript
    ("javascript", "has_property", "dynamic_typing"),
    ("javascript", "has_property", "prototype_inheritance"),
    ("javascript", "created_by", "brendan_eich"),
    ("javascript", "runs_in", "browser"),
    ("javascript", "runs_on", "nodejs"),
    # TypeScript
    ("typescript", "superset_of", "javascript"),
    ("typescript", "has_property", "static_typing"),
    ("typescript", "transpiles_to", "javascript"),
    ("typescript", "created_by", "microsoft"),
    # Ruby
    ("ruby_language", "has_property", "dynamic_typing"),
    ("ruby_language", "has_property", "object_oriented"),
    ("ruby_language", "created_by", "matz"),
    ("ruby_language", "philosophy", "developer_happiness"),
    # PHP
    ("php_language", "designed_for", "web_development"),
    ("php_language", "has_property", "dynamic_typing"),
    ("php_language", "runs_on", "server"),
    # Swift
    ("swift", "created_by", "apple"),
    ("swift", "successor_of", "objective_c"),
    ("swift", "has_property", "static_typing"),
    ("swift", "has_property", "reference_counted"),
    # Haskell
    ("haskell_language", "has_property", "purely_functional"),
    ("haskell_language", "has_property", "lazy_evaluation"),
    ("haskell_language", "has_property", "static_typing"),
    ("haskell_language", "uses", "type_inference"),
    # Erlang/Elixir
    ("erlang", "designed_for", "concurrency"),
    ("erlang", "uses", "actor_model"),
    ("erlang", "runs_on", "beam_vm"),
    ("elixir", "runs_on", "beam_vm"),
    ("elixir", "has_property", "functional"),
    # Clojure
    ("clojure", "is_a", "lisp_family"),
    ("clojure", "runs_on", "jvm"),
    ("clojure", "has_property", "functional"),
    ("clojure", "uses", "immutable_data"),
    # Scala
    ("scala_language", "runs_on", "jvm"),
    ("scala_language", "combines", "object_oriented"),
    ("scala_language", "combines", "functional"),
    # =========================================================================
    # LANGUAGE RELATIONSHIPS
    # =========================================================================
    # Influences
    ("c_language", "influenced", "cpp_language"),
    ("c_language", "influenced", "java_language"),
    ("c_language", "influenced", "csharp_language"),
    ("c_language", "influenced", "go"),
    ("c_language", "influenced", "rust_language"),
    ("smalltalk", "influenced", "ruby_language"),
    ("smalltalk", "influenced", "objective_c"),
    ("ml", "influenced", "haskell_language"),
    ("ml", "influenced", "ocaml"),
    ("ml", "influenced", "rust_language"),
    ("lisp_family", "influenced", "scheme"),
    ("lisp_family", "influenced", "clojure"),
    ("erlang", "influenced", "elixir"),
    # =========================================================================
    # TOOLING
    # =========================================================================
    # Compilers
    ("gcc", "compiles", "c_language"),
    ("gcc", "compiles", "cpp_language"),
    ("clang", "compiles", "c_language"),
    ("clang", "compiles", "cpp_language"),
    ("clang", "uses", "llvm"),
    ("rustc", "compiles", "rust_language"),
    ("rustc", "uses", "llvm"),
    ("ghc_compiler", "compiles", "haskell_language"),
    # Package managers
    ("npm", "manages", "javascript"),
    ("pip", "manages", "python_language"),
    ("cargo", "manages", "rust_language"),
    ("gem", "manages", "ruby_language"),
    ("maven", "manages", "java_language"),
    ("gradle", "manages", "java_language"),
    ("nuget", "manages", "csharp_language"),
    ("composer", "manages", "php_language"),
    # LSP
    ("lsp", "provides", "code_completion"),
    ("lsp", "provides", "diagnostics"),
    ("lsp", "provides", "go_to_definition"),
    ("pyright", "implements", "lsp"),
    ("rust_analyzer", "implements", "lsp"),
    # =========================================================================
    # PARADIGM RELATIONSHIPS
    # =========================================================================
    ("functional", "emphasizes", "immutability"),
    ("functional", "uses", "pure_functions"),
    ("object_oriented", "uses", "encapsulation"),
    ("object_oriented", "uses", "inheritance"),
    ("object_oriented", "uses", "polymorphism"),
    ("reactive", "uses", "observables"),
    ("concurrent", "handles", "parallelism"),
    # =========================================================================
    # TYPE SYSTEMS
    # =========================================================================
    ("static_typing", "catches_errors_at", "compile_time"),
    ("dynamic_typing", "checks_types_at", "runtime"),
    ("strong_typing", "prevents", "implicit_conversion"),
    ("duck_typing", "checks", "behavior"),
    ("gradual_typing", "allows", "optional_annotations"),
]


# =============================================================================
# LOADER FUNCTION
# =============================================================================


def load_web_into_core(core, agent_id: str = None) -> int:
    """Load all web and programming language knowledge into Dorian Core."""

    if agent_id is None:
        web_agent = core.register_agent("web_loader", domain="web", can_verify=True)
        agent_id = web_agent.agent_id

    count = 0

    # Load web categories
    print(f"  Loading {len(WEB_CATEGORIES)} web categories...")
    for name, parent, description in WEB_CATEGORIES:
        if core.ontology:
            from dorian_ontology import Category

            parent_cat = core.ontology.categories.get(parent)
            parent_level = parent_cat.level if parent_cat else 3
            core.ontology._add_category(
                Category(
                    name=name,
                    description=description,
                    parent=parent,
                    domain="web",
                    level=parent_level + 1,
                )
            )

        result = core.write(
            name,
            "subtype_of",
            parent,
            agent_id,
            source="web_ontology",
            check_contradictions=False,
        )
        if result.success:
            count += 1

    # Load language categories
    print(f"  Loading {len(LANG_CATEGORIES)} language categories...")
    for name, parent, description in LANG_CATEGORIES:
        if core.ontology:
            from dorian_ontology import Category

            parent_cat = core.ontology.categories.get(parent)
            parent_level = parent_cat.level if parent_cat else 3
            core.ontology._add_category(
                Category(
                    name=name,
                    description=description,
                    parent=parent,
                    domain="programming_languages",
                    level=parent_level + 1,
                )
            )

        result = core.write(
            name,
            "subtype_of",
            parent,
            agent_id,
            source="language_ontology",
            check_contradictions=False,
        )
        if result.success:
            count += 1

    # Load web facts
    print(f"  Loading {len(WEB_FACTS)} web facts...")
    for s, p, o in WEB_FACTS:
        result = core.write(
            s, p, o, agent_id, source="web_knowledge", check_contradictions=False
        )
        if result.success:
            count += 1

    # Load language facts
    print(f"  Loading {len(LANG_FACTS)} language facts...")
    for s, p, o in LANG_FACTS:
        result = core.write(
            s, p, o, agent_id, source="language_knowledge", check_contradictions=False
        )
        if result.success:
            count += 1

    print(f"  Total: {count} web/language facts loaded")
    return count


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("═" * 60)
    print("DORIAN WEB & PROGRAMMING LANGUAGES DOMAIN")
    print("═" * 60)

    print(f"\nWeb categories: {len(WEB_CATEGORIES)}")
    print(f"Language categories: {len(LANG_CATEGORIES)}")
    print(f"Web facts: {len(WEB_FACTS)}")
    print(f"Language facts: {len(LANG_FACTS)}")
    print(f"\nTotal categories: {len(WEB_CATEGORIES) + len(LANG_CATEGORIES)}")
    print(f"Total facts: {len(WEB_FACTS) + len(LANG_FACTS)}")

    print("\nSample web categories:")
    for name, parent, desc in WEB_CATEGORIES[:10]:
        print(f"  {name} <- {parent}")

    print("\nSample language categories:")
    for name, parent, desc in LANG_CATEGORIES[:10]:
        print(f"  {name} <- {parent}")

    print("\nSample web facts:")
    for s, p, o in WEB_FACTS[:10]:
        print(f"  {s} {p} {o}")

    print("\nSample language facts:")
    for s, p, o in LANG_FACTS[:10]:
        print(f"  {s} {p} {o}")
