// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Parts of the code are under this licence

import Flutter
import UIKit
import TensorFlowLite

public class SwiftYolov4tflitePlugin: NSObject, FlutterPlugin {
  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(name: "yolov4tflite", binaryMessenger: registrar.messenger())
    let instance = SwiftYolov4tflitePlugin()
    registrar.addMethodCallDelegate(instance, channel: channel)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    if(call.methode == "getPlatformVersion"){
        result("iOS " + UIDevice.current.systemVersion)
    }
    
  }
}

public class Yolov4Classifier{
    // MARK: - Internal Properties
    /// The current thread count used by the TensorFlow Lite Interpreter.
    let threadCount: Int

    let resultCount = 3
    let threadCountLimit = 10

    let threshold: Float = 0.5

    // MARK: - Model Parameters
    let batchSize = 1
    let inputChannels = 3
    let inputSize : Int
    let output_width : Int
    let isTiny : Bool

    // image mean and std for floating model, should be consistent with parameters used in model training
    let imageMean: Float = 127.5
    let imageStd:  Float = 127.5

    // MARK: - Private Properties
    /// List of labels from the given labels file.
    private var labels: [String] = []

    /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
    private var interpreter: Interpreter

    /// Information about the alpha component in RGBA data.
    private let alphaComponent = (baseOffset: 4, moduloRemainder: 3)

    // MARK: - Initialization
    /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
    /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
    init?(modelFileInfo: FileInfo, labelData: String, threadCount: Int = 1,inputSize: Int,isTiny : Bool) {
        let modelFilename = modelFileInfo.name

        // Construct the path to the model file.
        guard let modelPath = Bundle.main.path(
        forResource: modelFilename,
        ofType: modelFileInfo.extension
        ) else {
        print("Failed to load the model file with name: \(modelFilename).")
        return nil
        }

        // Set model size
        self.inputSize = inputSize
        self.isTiny = isTiny
        if(isTiny){
            self.output_width = 2535
        }else{
            self.output_width = 10647
        }

        // Specify the options for the `Interpreter`.
        self.threadCount = threadCount
        var options = InterpreterOptions()
        options.threadCount = threadCount
        do {
        // Create the `Interpreter`.
        interpreter = try Interpreter(modelPath: modelPath, options: options)
        // Allocate memory for the model's input `Tensor`s.
        try interpreter.allocateTensors()
        } catch let error {
        print("Failed to create the interpreter with error: \(error.localizedDescription)")
        return nil
        }
        // Load the classes listed in the labels file.
        loadLabels(labelData: String)
    }

    private loadLabels(labelData: String){
        labels = split(labelData,{$0 == ","})
    }

    /// This class handles all data preprocessing and makes calls to run inference on a given frame
    /// through the `Interpeter`. It then formats the inferences obtained and returns the top N
    /// results for a successful inference.
    func runModel(onFrame pixelBuffer: CVPixelBuffer) -> [Recognition]? {
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
                sourcePixelFormat == kCVPixelFormatType_32BGRA ||
                sourcePixelFormat == kCVPixelFormatType_32RGBA)


        let imageChannels = 4
        assert(imageChannels >= inputChannels)

        // Crops the image to the biggest square in the center and scales it down to model dimensions.
        let scaledSize = CGSize(width: inputWidth, height: inputHeight)
        guard let scaledPixelBuffer = pixelBuffer.resized(to: scaledSize) else {
        return nil
        }

        let interval: TimeInterval
        let outputBoundingBox: Tensor
        let outputScores: Tensor
        do {
        let inputTensor = try interpreter.input(at: 0)

        // Remove the alpha component from the image buffer to get the RGB data.
        guard let rgbData = rgbDataFromBuffer(
            scaledPixelBuffer,
            byteCount: batchSize * inputWidth * inputHeight * inputChannels,
            isModelQuantized: inputTensor.dataType == .uInt8
        ) else {
            print("Failed to convert the image buffer to RGB data.")
            return nil
        }

        // Copy the RGB data to the input `Tensor`.
        try interpreter.copy(rgbData, toInputAt: 0)

        // Run inference by invoking the `Interpreter`.
        let startDate = Date()
        try interpreter.invoke()
        interval = Date().timeIntervalSince(startDate) * 1000

        outputBoundingBox = try interpreter.output(at: 0)
        outputScores = try interpreter.output(at: 1)
        } catch let error {
        print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
        return nil
        }

        // Formats the results
        let resultArray = formatResultsFull(
                boundingBox: [Float](unsafeData: outputBoundingBox.data) ?? [],
                outputScores: [Float](unsafeData: outputScores.data) ?? [],
                width: CGFloat(imageWidth),
                height: CGFloat(imageHeight))

        return nms(resultArray)
    }

    
    func formatResults(boundingBox: [Float], outputScores: [Float], width: CGFloat, height: CGFloat) -> [Recognition]{
        var detections: [Recognition] = []
        
        let gridWidth:Int = output_width[0]

        for i in 0..gridWidth{
            let maxClass :Float = 0;
            let detectedClass :Int = -1;
            let classes: Float[] = new Float[labels.size()];
            for c in 0..labels.size(){
                classes [c] = outputScores[0][i][c];
            }
            for c in 0..labels.size(){
                if (classes[c] > maxClass){
                    detectedClass = c;
                    maxClass = classes[c];
                }
            }
            let score . Float = maxClass;
            if (score > threshold){
                let xPos :Float = bboxes[0][i][0];
                let yPos :Float = bboxes[0][i][1];
                let w: Float = bboxes[0][i][2];
                let h :Float = bboxes[0][i][3];
                // miss use
                let rectF = CGRect(
                        x: Math.max(0, xPos - w / 2),
                        y: Math.max(0, yPos - h / 2),
                        w: Math.min(width - 1, xPos + w / 2),
                        h: Math.min(height - 1, yPos + h / 2));
                detections.add(Recognition("" + i, labels.get(detectedClass),score,rectF,detectedClass ));
            }
        }

        return detections
    }


    /// Loads the labels from the labels file and stores them in the `labels` property.
    private func loadLabels(fileInfo: FileInfo) {
        let filename = fileInfo.name
        let fileExtension = fileInfo.extension
        guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
        fatalError("Labels file not found in bundle. Please add a labels file with name " +
                    "\(filename).\(fileExtension) and try again.")
        }
        do {
        let contents = try String(contentsOf: fileURL, encoding: .utf8)
        labels = contents.components(separatedBy: .newlines)
        } catch {
        fatalError("Labels file named \(filename).\(fileExtension) cannot be read. Please add a " +
                    "valid labels file and try again.")
        }
    }

    /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
    ///
    /// - Parameters
    ///   - buffer: The BGRA pixel buffer to convert to RGB data.
    ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
    ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
    ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
    ///       floating point values).
    /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
    ///     converted.
    private func rgbDataFromBuffer(
        _ buffer: CVPixelBuffer,
        byteCount: Int,
        isModelQuantized: Bool
    ) -> Data? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer {
        CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
        }
        guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
        return nil
        }
        
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let destinationChannelCount = 3
        let destinationBytesPerRow = destinationChannelCount * width
        
        var sourceBuffer = vImage_Buffer(data: sourceData,
                                        height: vImagePixelCount(height),
                                        width: vImagePixelCount(width),
                                        rowBytes: sourceBytesPerRow)
        
        guard let destinationData = malloc(height * destinationBytesPerRow) else {
        print("Error: out of memory")
        return nil
        }
        
        defer {
        free(destinationData)
        }

        var destinationBuffer = vImage_Buffer(data: destinationData,
                                            height: vImagePixelCount(height),
                                            width: vImagePixelCount(width),
                                            rowBytes: destinationBytesPerRow)
        
        if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA){
        vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        } else if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32ARGB) {
        vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        }

        let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
        if isModelQuantized {
        return byteData
        }

        // Not quantized, convert to floats
        let bytes = Array<UInt8>(unsafeData: byteData)!
        var floats = [Float]()
        for i in 0..<bytes.count {
        floats.append((Float(bytes[i]) - imageMean) / imageStd)
        }
        return Data(copyingBufferOf: floats)
    }

    func nms(list: Recognition[]) -> [Recognition]{
        let nmsList: Recognition[] = [];

        for k in 0..labels.size(){
            //1.find max confidence per class
            PriorityQueue<Recognition> pq =
                    new PriorityQueue<Recognition>(
                            50,
                            new Comparator<Recognition>() {
                                @Override
                                public int compare(final Recognition lhs, final Recognition rhs) {
                                    // Intentionally reversed to put high confidence at the head of the queue.
                                    return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                                }
                            });

            for (int i = 0; i < list.size(); ++i) {
                if (list.get(i).getDetectedClass() == k) {
                    pq.add(list.get(i));
                }
            }

            //2.do non maximum suppression
            while (pq.size() > 0) {
                //insert detection with max confidence
                Recognition[] a = new Recognition[pq.size()];
                Recognition[] detections = pq.toArray(a);
                Recognition max = detections[0];
                nmsList.add(max);
                pq.clear();

                for (int j = 1; j < detections.length; j++) {
                    Recognition detection = detections[j];
                    RectF b = detection.getLocation();
                    if (box_iou(max.getLocation(), b) < mNmsThresh) {
                        pq.add(detection);
                    }
                }
            }
        }
        return nmsList;
    }

    protected float mNmsThresh = 0.6f;

    protected float box_iou(RectF a, RectF b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    protected float box_intersection(RectF a, RectF b) {
        float w = overlap((a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left);
        float h = overlap((a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top);
        if (w < 0 || h < 0) return 0;
        float area = w * h;
        return area;
    }

    protected float box_union(RectF a, RectF b) {
        float i = box_intersection(a, b);
        float u = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i;
        return u;
    }

    protected float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
    }
}




public class Recognition : CoustomStringConvertibale{
    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of
     * the object.
     */
    let id : String;

    /**
     * Display name for the recognition.
     */
    let title : String;

    /**
     * A sortable score for how good the recognition is relative to others. Higher should be better.
     */
    let confidence: Float;

    /**
     * Optional location within the source image for the location of the recognized object.
     */
    let location: CGRect;

    let detectedClass: Int;
/*
    public Recognition(
            final String id, final String title, final Float confidence, final RectF location) {
        this.id = id;
        this.title = title;
        this.confidence = confidence;
        this.location = location;
    }
*/
    public Recognition(id:String,title:String,confidence: Float,location:CGRect, detectedClass:Int) {
        this.id = id;
        this.title = title;
        this.confidence = confidence;
        this.location = location;
        this.detectedClass = detectedClass;
    }
/*
    public String getId() {
        return id;
    }

    public String getTitle() {
        return title;
    }

    public Float getConfidence() {
        return confidence;
    }

    public RectF getLocation() {
        return new RectF(location);
    }

    public void setLocation(RectF location) {
        this.location = location;
    }

    public int getDetectedClass() {
        return detectedClass;
    }

    public void setDetectedClass(int detectedClass) {
        this.detectedClass = detectedClass;
    }
 */

    public var description: String{
        let resultString : String = "{";
        if (id != null) {
            resultString += "\"id\": " + id + ",";
        }

        if (title != null) {
            resultString += "\"detectedClass\": \"" + title + "\",";
        }

        if (confidence != null) {
            resultString += "\"confidenceInClass\": " +  confidence +",";
        }

        if (location != null) {
            resultString += "\"rect\": { \"t\": " + location.x + ", \"b\": " + location.h +", \"l\": " + location.y + ", \"r\": " + location.w + "}";
        }

        resultString += "}";

        return resultString.trim();
    }
}
