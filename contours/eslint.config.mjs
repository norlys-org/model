//@ts-nocheck
import globals from "globals";
import tsParser from "@typescript-eslint/parser";
import lemonPledge from 'lemon-pledge'

export default [
  lemonPledge.configs['typed-react'],

  {
    ignores: [
      '.next',
      'next.config.mjs',
      'eslint.config.mjs',
      'public',
      'app/helpers/solar.tsx',
      'app/helpers/d3Legend.ts'
    ],
  },

  {
    languageOptions: {
      globals: {
        ...globals.browser,
      },

      parser: tsParser,
      ecmaVersion: 12,
      sourceType: "module",

      parserOptions: {
        ecmaFeatures: {
          jsx: true,
        },

        project: "./tsconfig.json",
      },
    },
  },

  {
    files: ["**/*.ts", "**/*.tsx"],

    languageOptions: {
      ecmaVersion: 5,
      sourceType: "script",

      parserOptions: {
        project: "./tsconfig.json",
      },
    },
  }
]
