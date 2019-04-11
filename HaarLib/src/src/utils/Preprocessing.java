/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package src.utils;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 *
 * @author Andika Mulyawan
 */
public final class Preprocessing {

    /**
     * ######################################################################
     * Method untuk memperoleh fitur gambar dalam bentuk data satu dimensi.
     *
     */
    public static Mat getEdge(Mat frame, double width, double height, double treshold) {
        Imgproc.resize(frame, frame, new Size(width, height));
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(frame, frame, new Size(7.0, 7.0), 0.0);
        Imgproc.Canny(frame, frame, treshold, treshold * 3);
        frame = frame.reshape(1, 1);
        return frame;
    }

    /**
     * ######################################################################
     * Method untuk mengambil gambar dalam ROI.
     */
    public static Mat getBox(Mat frame) {
        Rect rectCrop = new Rect(
                new Point(frame.cols() - 5, 10 + 5),
                new Point(frame.cols() / 2 + 5, frame.rows() - (frame.rows() / 3) - 10 - 5));
        frame = frame.submat(rectCrop);
        return frame;
    }

    /**
     * ######################################################################
     * Method untuk menggambar kotak ROI pada gambar.
     */
    public static Mat drawRect(Mat frame) {
        Imgproc.rectangle(frame,
                new Point(frame.cols(), 10),
                new Point(frame.cols() / 2, frame.rows() - (frame.rows() / 3) - 10),
                new Scalar(0, 0, 255),
                3);
        return frame;
    }

    /**
     * ######################################################################
     * Method untuk menghitung nilai tinggi gambar dengan perbandingan lebar
     *
     * @param data
     * @return
     */
    public static double getHeight(double widhtCompr, double widthOri, double heightOri) {
        heightOri = heightOri * (widhtCompr / widthOri);
        return heightOri;
    }

}
