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
    let detector: Yolov4Classifier
    var registrar: FlutterPluginRegistrar? = nil

  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(name: "yolov4tflite", binaryMessenger: registrar.messenger())
    let instance = SwiftYolov4tflitePlugin()
    registrar.addMethodCallDelegate(instance, channel: channel)
    instance.registrar = registrar
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    if(call.methode == "getPlatformVersion"){
        result("iOS " + UIDevice.current.systemVersion)
    }else if(call.methode == "loadModel"){
        let modelPath = call.argument("model")
        let labels = call.argument("labels")
        let key = registrar?.lookupKey(forAsset: modelPath)

        loadModel(labelData: labels,modelFileKey: key,result: result)
    }else if(call.methode = "detectObjects"){
        let path = call.methode("image")

        detectObjects(imagePath: path,result: result)
    }
  }

  func loadModel(labelData: String,modelFileKey: String,result: @escaping FlutterResult){
        DispatchQueue.global(qos: .userInitiated).async {
            do{
                detector = try Yolov4Classifier(labelData: labelData,modelFileKey: modelFileKey)
                DispatchQueue.main.sync{
                    result("Succsess")
                }
            }catch let error{
                DispatchQueue.main.sync{
                    result(error.localizedDescription)
                }
            }
        }
    }

    func detectObjects(imagePath:String, result: @escaping FlutterResult){
        DispatchQueue.global(qos: .userInitiated).async {
            do{
                let imageData = try Data(contentsOf: fileURL)
                let image = UIImage(data: imageData)
                let pixelBuffer = buffer(image: image)
                
                let recognitions = try detector.runModel(pixelBuffer: pixelBuffer)

                var reg : [String] = []
                for r in recognitions{
                    reg.append(String(r))
                }

                DispatchQueue.main.sync{
                    result(reg)
                }
            }catch let error{
                DispatchQueue.main.sync{
                    result(error.localizedDescription)
                }
            }
        }
    }

    func buffer(from image: UIImage) -> CVPixelBuffer? {
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer : CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(image.size.width), Int(image.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        guard (status == kCVReturnSuccess) else {
            return nil
        }

        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)

        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData, width: Int(image.size.width), height: Int(image.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

        context?.translateBy(x: 0, y: image.size.height)
        context?.scaleBy(x: 1.0, y: -1.0)

        UIGraphicsPushContext(context!)
        image.draw(in: CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))

        return pixelBuffer
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
    init?(modelFileKey: String, labelData: String, threadCount: Int = 1,inputSize: Int = 416,isTiny : Bool = true) {


        // Construct the path to the model file.
        guard let modelPath = Bundle.main.path(
        forResource: modelFileKey,
        ofType: "tflite"
        ) else {
        print("Failed to load the model file with name: \(modelFileKey).")
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

    func loadLabels(labelData: String){
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
            let classes: [Float] = new Float[labels.size()];
            for c in 0..labels.size(){
                classes [c] = outputScores[0][i][c];
            }
            for c in 0..labels.size(){
                if (classes[c] > maxClass){
                    detectedClass = c;
                    maxClass = classes[c];
                }
            }
            let score : Float = maxClass;
            if (score > threshold){
                let xPos :Float = bboxes[0][i][0];
                let yPos :Float = bboxes[0][i][1];
                let w: Float = bboxes[0][i][2];
                let h :Float = bboxes[0][i][3];
                // miss use
                let rectF = CGRect(
                        x: Math.max(0, xPos - w / 2), // left
                        y: Math.max(0, yPos - h / 2), // top
                        w: Math.min(width - 1, xPos + w / 2),  // rigth
                        h: Math.min(height - 1, yPos + h / 2)); // bottom
                detections.add(Recognition("" + i, labels.get(detectedClass),score,rectF,detectedClass ));
            }
        }

        return detections
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

    func compare(first lhs: Recognition,second rhs Recognition) -> Bool{
        return lhs.getConfidence().isLess(then: rhs.getConfidence())
    }
    
    func nms(list: [Recognition]) -> [Recognition]{
        let nmsList: [Recognition] = [];

        for k in 0..labels.size(){
            //1.find max confidence per class
            pq: Heap<Recognition> = new Heap<Recognition>(priorityFunction: compare)

            for i in 0..list.size(){
                if (list.get(i).getDetectedClass() == k) {
                    pq.enqueue(list.get(i));
                }
            }

            //2.do non maximum suppression
            while (!pq.isEmpty) {
                //insert detection with max confidence
                a = new Recognition[pq.count];
                detections = pq.toArray(a);
                max = detections[0];
                nmsList.add(max);
                pq.clear();

                for j in 0..detections.count {
                    detection = detections[j];
                    b = detection.getLocation();
                    if (box_iou(max.getLocation(), b) < mNmsThresh) {
                        pq.enqueue(detection);
                    }
                }
            }
        }
        return nmsList
    }

    let mNmsThresh: Float = 0.6

    func box_iou(a : CGRect, b : CGRect) -> Float {
        return box_intersection(a, b) / box_union(a, b)
    }

    func box_intersection(a: CGRect, b: CGRect) -> Float {
        let w : Float = overlap((a.x + a.w) / 2, a.w - a.x,
                (b.x + b.w) / 2, b.w - b.x)
        let h : Float = overlap((a.y + a.h) / 2, a.h - a.y,
                (b.y + b.h) / 2, b.h - b.y)
        if (w < 0 || h < 0) {return 0}
        float area = w * h
        return area
    }

    func box_union(a: CGRect, b: CGRect) -> Float {
        let i = box_intersection(a, b)
        let u = (a.w - a.x) * (a.h - a.y) + (b.w - b.x) * (b.h - b.y) - i
        return u
    }

    func overlap(x1 : Float,w1: Float, x2: Float, w2: Float) -> Float{
        let l1 = x1 - w1 / 2;
        let l2 = x2 - w2 / 2;
        let left = l1 > l2 ? l1 : l2;
        let r1 = x1 + w1 / 2;
        let r2 = x2 + w2 / 2;
        let right = r1 < r2 ? r1 : r2;
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
    init?(id:String,title:String,confidence: Float,location:CGRect, detectedClass:Int) {
        self.id = id;
        self.title = title;
        self.confidence = confidence;
        self.location = location;
        self.detectedClass = detectedClass;
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
     */
    func getLocation() -> CGRect{
        return location;
    }

    func setLocation(location: CGRect) {
        self.location = location;
    }

    func setDetectedClass(detectedClass: Int) {
        self.detectedClass = detectedClass;
    }
 
    func getDetectedClass() -> Int {
        return detectedClass;
    }

    public var description: String{
        let resultString : String = "{"
        if (id != null) {
            resultString += "\"id\": " + id + ","
        }

        if (title != null) {
            resultString += "\"detectedClass\": \"" + title + "\","
        }

        if (confidence != null) {
            resultString += "\"confidenceInClass\": " +  confidence + ","
        }

        if (location != null) {
            resultString += "\"rect\": { \"t\": " + location.x + ", \"b\": " + location.h + ", \"l\": " + location.y + ", \"r\": " + location.w + "}"
        }

        resultString += "}"

        return resultString.trim()
    }
}


// designd after https://www.raywenderlich.com/586-swift-algorithm-club-heap-and-priority-queue-data-structure
struct Heap<Recognition> {
    var elements: [Recognition]
    let priorityFunction: (Recognition,Recognition) -> Bool
    
    var isEmpty : Bool{
        return elements.isEmpty
    }
    
    var count : Int {
        return elements.count
    }
    
    init(elements: [Recognition] = [],priorityFunction: @escaping (Recognition,Recognition)->Bool) {
        self.elements = elements
        self.priorityFunction = priorityFunction
        buildHeap()
    }
    
    mutating func buildHeap(){
        for index in (0..< count / 2).reversed(){
            siftDown(elementAtIndex: index)
        }
    }
    
    func peek() -> Recognition? {
        return elements.first
    }
    
    func toArray() -> [Recognition]{
        return elements
    }
    
    func clear(){
        self.elements = []
    }
    
    mutating func enqueue(_ element: Recognition){
        elements.append(element)
        siftUp(elementAtIndex: count - 1)
    }
    
    mutating func siftUp(elementAtIndex index: Int){
        let parant = parentIndex(of: index)
        guard !isRoot(index), isHigherPriority(at: index,than: parant) else {
            return
        }
        swapElement(at index,with: parant)
        siftUp(elementAtIndex: parant)
    }
    
    mutating func dequeue() -> Recognition? {
        guard !isEmpty else {
            return nil
        }
        swapElement(at: 0,with: count - 1)
        let element = elements.removeLast()
        if !isEmpty {
            siftDown(elementAtIndex: 0)
        }
        return element
    }
    
    mutating func siftDown(elementAtIndex index :Int){
        let childIndex = heighestPriorityIndex(for: index)
        if index == childIndex{
            return
        }
        swapElement(at: index,with: childIndex)
        siftDown(elementAtIndex: childIndex)
    }
    
    func isRoot(_ index: Int)-> Bool {
        return (index == 0)
    }
    
    func leftChildIndex(of index: Int) -> Int{
        return (2* index) + 1
    }
    
    func rigthChildIndex(of index: Int) -> Int{
        return (2* index) + 2
    }
    
    func parentIndex(of index : Int)->Int{
        return (index - 1) / 2
    }
    
    func isHigherPriority(at firstIndex: Int, than secondIndex: Int)-> Bool {
        return priorityFunction(elements[firstIndex],elements[secondIndex])
    }
    
    func heighestPriorityIndex(of parantIndex: Int, and childIndex: Int) -> {
        guard childIndex < count && isHigherPriority(at: childIndex, than: parentIndex) else {
            return parantIndex
        }
        return childIndex
    }
    
    func heighestPriorityIndex(for parant: Int) -> int{
        return heighestPriorityIndex(of: heighestPriorityIndex(of: parant,and leftChildIndex(of: parant)),and rigthChildIndex(of: parant))
    }
    
    mutating func swapElement(at firstIndex: Int, with secondIndex: Int){
        guard firstIndex != secondIndex else {
            return
        }
        swap(&elements[firstIndex],&elements[secondIndex])
    }
    
    
}
