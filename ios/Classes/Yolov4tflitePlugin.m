#import "Yolov4tflitePlugin.h"
#if __has_include(<yolov4tflite/yolov4tflite-Swift.h>)
#import <yolov4tflite/yolov4tflite-Swift.h>
#else
// Support project import fallback if the generated compatibility header
// is not copied when this plugin is created as a library.
// https://forums.swift.org/t/swift-static-libraries-dont-copy-generated-objective-c-header/19816
#import "yolov4tflite-Swift.h"
#endif

@implementation Yolov4tflitePlugin
+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
  [SwiftYolov4tflitePlugin registerWithRegistrar:registrar];
}
@end
