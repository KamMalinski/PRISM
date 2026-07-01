name: Build macOS

on:
  workflow_dispatch:
  push:
    tags:
      - "v*"

jobs:
  build-macos:
    runs-on: macos-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Build PRISM
        run: |
          python3 --version
          rm -rf .venv-build build dist
          bash scripts/build/build_macos.sh

      - name: Smoke test macOS build
        run: |
          ./dist/PRISM.app/Contents/MacOS/PRISM --ui-smoke-test

      - name: Package macOS build
        run: |
          ditto -c -k --sequesterRsrc --keepParent dist/PRISM.app PRISM-macos.zip

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: PRISM-macos
          path: PRISM-macos.zip
          if-no-files-found: error
