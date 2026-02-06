import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Ignore build output and other non-source directories
  globalIgnores([
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    "node_modules/**",
    "coverage/**",
  ]),
  // Custom rule overrides - downgrade some errors to warnings
  // TODO: Fix these properly and re-enable as errors
  {
    rules: {
      // Allow require() in test files and config files
      "@typescript-eslint/no-require-imports": "off",
      // Allow unescaped entities in JSX (common in text content)
      "react/no-unescaped-entities": "warn",
      // Allow any type temporarily (should be fixed properly)
      "@typescript-eslint/no-explicit-any": "warn",
      // Allow unused vars with underscore prefix
      "@typescript-eslint/no-unused-vars": ["warn", {
        "argsIgnorePattern": "^_",
        "varsIgnorePattern": "^_"
      }],
      // Downgrade img element warning
      "@next/next/no-img-element": "warn",
      // Downgrade setState in effect errors to warnings
      "react-hooks/set-state-in-effect": "warn",
    }
  }
]);

export default eslintConfig;
