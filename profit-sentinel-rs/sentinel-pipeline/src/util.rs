/// Extract a short type name from the full module path.
///
/// Given `"my_crate::some_module::MyType"`, returns `"MyType"`.
pub fn short_type_name(full: &str) -> &str {
    full.rsplit("::").next().unwrap_or(full)
}
