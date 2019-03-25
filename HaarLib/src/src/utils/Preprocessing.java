/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package src.utils;

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
     * method memperoleh garis tepi pada gambar
     * method akan mengembalikan gambar berisi garis tapi
     * var:
     * Mat frame : berisi gambar awal
     */
    public static Mat getEdgeView(Mat frame, double width, double height, double treshold) {

        Imgproc.resize(frame, frame, new Size(width, height));
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(frame, frame, new Size(7.0, 7.0), 0.0);
        Imgproc.Canny(frame, frame, treshold, treshold * 3);
        return frame;
    }

    /**
     * ######################################################################
     * method memperoleh garis tepi pada gambar
     * gambar yang diperoleh akan dilakukan resaping agar diperoleh fitur
     * method akan mengembalikan nilai fitur hasil reshaping
     * var:
     * Mat frame : berisi gambar awal
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
     * method memperoleh gambar dalam ROI kotak merah
     * method akan mengembalikan gambar setalah dilakukan bluring
     * var:
     * Mat frame : berisi gambar awal dengan dari kamera
     * Rect rectCrop : berisi koordinat ROI kotak merah
     * 12/6/18 6:59 AM
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
     * method memperoleh gambar dalam ROI kotak merah
     * method akan mengembalikan gambar setalah dilakukan bluring
     * var:
     * Mat frame : berisi gambar awal dengan dari kamera
     * Rect rectCrop : berisi koordinat ROI kotak merah
     * 12/6/18 6:59 AM
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
     *
     * @param data
     * @return
     */
    public static double getHeight(double widhtCompr, double widthOri, double heightOri) {
        heightOri = heightOri * (widhtCompr / widthOri);
        return heightOri;
    }

}
