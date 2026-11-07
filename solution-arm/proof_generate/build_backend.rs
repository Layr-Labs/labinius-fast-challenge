/// A native build may use host features omitted from LLVM's target feature list.
/// Cross builds rely exclusively on their explicit target features.
pub fn use_bdep(features: &str, flags: &str, host: &str, target: &str, detected: bool) -> bool {
    if features.split(',').any(|f| f == "sve2-bitperm") {
        return true;
    }
    if host != target || !detected || !features.split(',').any(|f| f == "sve2") {
        return false;
    }
    let mut cpu = None;
    let mut disabled = false;
    let mut args = flags.split('\x1f');
    while let Some(arg) = args.next() {
        let option = if arg == "-C" { args.next().unwrap_or("") } else { arg.strip_prefix("-C").unwrap_or("") };
        if let Some(value) = option.strip_prefix("target-cpu=") {
            cpu = Some(value);
        }
        if let Some(value) = option.strip_prefix("target-feature=") {
            for feature in value.split(',') {
                match feature {
                    "-sve2-bitperm" => disabled = true,
                    "+sve2-bitperm" => disabled = false,
                    _ => {}
                }
            }
        }
    }
    cpu == Some("native") && !disabled
}

#[cfg(test)]
mod tests {
    use super::use_bdep;
    const ARM: &str = "aarch64-unknown-linux-gnu";
    const NATIVE: &str = "-C\x1ftarget-cpu=native";

    #[test]
    fn native_hardware_probe_fills_missing_compiler_feature() {
        assert!(use_bdep("neon,sve,sve2", NATIVE, ARM, ARM, true));
        assert!(use_bdep("sve2", "-Ctarget-cpu=native", ARM, ARM, true));
    }

    #[test]
    fn host_features_do_not_leak_into_cross_or_generic_builds() {
        assert!(!use_bdep("sve2", NATIVE, "x86_64-unknown-linux-gnu", ARM, true));
        assert!(!use_bdep("sve2", "-Ctarget-cpu=neoverse-v2", ARM, ARM, true));
        assert!(!use_bdep("sve2", "", ARM, ARM, true));
        assert!(!use_bdep("sve2", "-Ctarget-cpu=native\x1f-Ctarget-cpu=generic", ARM, ARM, true));
    }

    #[test]
    fn unavailable_or_disabled_features_keep_the_fallback() {
        assert!(!use_bdep("sve2", NATIVE, ARM, ARM, false));
        assert!(!use_bdep("neon", NATIVE, ARM, ARM, true));
        assert!(!use_bdep("sve2", "-Ctarget-cpu=native\x1f-Ctarget-feature=-sve2-bitperm", ARM, ARM, true));
    }

    #[test]
    fn explicit_target_feature_works_for_cross_builds() {
        assert!(use_bdep("sve2,sve2-bitperm", "", "x86_64-unknown-linux-gnu", ARM, false));
    }
}
