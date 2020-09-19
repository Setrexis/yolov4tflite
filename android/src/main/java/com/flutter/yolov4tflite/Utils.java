package com.flutter.yolov4tflite;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;

public class Utils {
    public static Bitmap processBitmap(Bitmap source, int size){

        int image_height = source.getHeight();
        int image_width = source.getWidth();

        Bitmap croppedBitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);

        return croppedBitmap;
    }
}
