# True Headless Environment Ruleset

## UNIVERSAL INITIALIZATION PATTERNS
- Default: ij = imagej.init() # Latest ImageJ2, headless mode
- Specific version: ij = imagej.init('2.14.0') 
- With Fiji: ij = imagej.init('sc.fiji:fiji')
- Local install: ij = imagej.init('/path/to/Fiji.app')

## SERVER/CLUSTER ENVIRONMENTS
- Always use mode='headless'
- No display capabilities - all output via files
- Use batch processing patterns
- Implement proper logging

## LIMITATIONS
- RoiManager: Limited functionality, no GUI operations
- WindowManager: Basic operations only
- No interactive plugins
- All visualization must be saved to files

## OPTIMIZATION
- Pre-download dependencies to avoid network issues
- Use local Fiji installations when possible
- Implement progress tracking for long operations
- Save intermediate results for debugging