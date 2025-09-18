# Interactive Environment Ruleset

## INITIALIZATION MODES (combine with version patterns)
- Full ImageJ UI, blocks current thread: ij = imagej.init(mode='gui')
- Graphical enabled, non-blocking init: ij = imagej.init(mode='interactive')
- In interactive mode, show UI with ij.ui().showUI()

## REPRODUCBILE VERSION PATTERNS
- Specific ImageJ2 version: ij = imagej.init('2.14.0') 
- Specific Fiji with ImageJ 1.x: imagej.init('sc.fiji:fiji:2.14.0')
- Maven endpoints are additive: imagej.init(['net.imagej:imagej:2.14.0', 'net.preibisch:BigStitcher:0.4.1'])

## NOT-REPRODUCIBLE VERSION PATTERNS
- Latest ImageJ2: ij = imagej.init()
- Latest Fiji with ImageJ 1.x: imagej.init('sc.fiji:fiji')
- Local install: ij = imagej.init('/path/to/Fiji.app')
- Maven endpoints are additive: imagej.init(['net.imagej:imagej', 'net.preibisch:BigStitcher'])

## GUI CAPABILITIES
- Display images: Both ij.py.show() and ij.ui().show() work
- Full RoiManager functionality
- WindowManager with all features
- Interactive plugin dialogs
- Real-time image display

## PLATFORM CONSIDERATIONS
- macOS: May require PyObjC for GUI mode
- Linux: Ensure X11 forwarding if remote
- Windows: Generally works out of box

## BEST PRACTICES
- Test in headless first for portability
- Use GUI for development/debugging
- Save processing scripts for batch use