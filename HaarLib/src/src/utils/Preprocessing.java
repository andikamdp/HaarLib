/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package src.utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
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
//######################################################################
/////////////
//method untuk menghilangkan titk hitam dan putih yang belum bersih dari hasil treshold
/////////////

    public static Mat cleaning(Mat frame) {
        Mat kernel = new Mat(new Size(3, 3), CvType.CV_16S, new Scalar(255));
//        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_OPEN, new Size(11, 11));

        Imgproc.morphologyEx(frame, frame, Imgproc.MORPH_CLOSE, kernel);
        Imgproc.morphologyEx(frame, frame, Imgproc.MORPH_OPEN, kernel);
//
//        Imgproc.dilate(frame, frame, kernel);
        Imgproc.dilate(frame, frame, kernel, new Point(0, 0), 2);
//        Imgproc.erode(frame, frame, kernel, new Point(0, 0), 1);

        return frame;
    }
//######################################################################
//######################################################################

//######################################################################
/////////////
//method untuk periksa arah titikY
/////////////
//if (titikA.x <= titikB.x)
//    jika titik pertama lebih keatas dari titik kedua
//31/08/2018
/////////////
    public static Boolean arahTitikY(Point titikA, Point titikB) {
        if (titikA.y < titikB.y) {
//            System.out.print(" Puncak->menurun");
//            System.out.println(titikA.toString() + " " + titikB.toString());
            return true;
        } else if (titikA.y == titikB.y) {
//            System.out.print(" sama");
            return false;
        } else {

//            System.out.println(titikA.toString() + " " + titikB.toString());
//            System.out.print(" Lembah->menaik");
            return false;
        }
    }
//######################################################################
//######################################################################

//######################################################################
/////////////
//method untuk periksa arah titikX
/////////////
//if (titikA.x <= titikB.x)
//    jika titik pertama lebih kekanan dari titik kedua
//31/08/2018
/////////////
    public static Boolean arauTitikX(Point titikA, Point titikB) {
        if (titikA.x > titikB.x) {
            return true;
        } else {
            return false;
        }
    }
//######################################################################
//######################################################################

//######################################################################
/////////////
//method untuk menghitung jarak titik antara index pada contour
/////////////
    public static void hitungJarakTitik(List<MatOfPoint> contour) {
        Point[] point = contour.get(0).toArray();
        for (int j = 0; j < point.length - 1; j++) {
            double jarak = Math.sqrt(Math.pow(point[j].x - point[j + 1].x, 2)
                    + Math.
                            pow(
                                    point[j].y
                                    - point[j + 1].y, 2));
        }
    }
//######################################################################
/////////////
//method untuk menghitung jarak titik sesuai index pada contour
/////////////

    public static void hitungJarakTitik(List<MatOfPoint> contour, Integer[] index) {
        Point[] point = contour.get(0).toArray();
        int k = 0;
        for (int j = 1; j < point.length - 1; j++) {
            if (index[j] != null && index[j] < 0) {
                double jarak = Math.sqrt(Math.
                        pow(point[k].x - point[j].x, 2)
                        + Math.pow(point[k].y - point[j].y, 2));
                k = j;
            }
        }
    }
//######################################################################
/////////////
//method untuk  menghitung jarak dua titik
/////////////

    public static double hitungJarakTitik(Point titikA, Point titikB) {

        double jarak = Math.sqrt(Math.pow(titikA.x - titikB.x, 2)
                + Math.pow(titikA.y - titikB.y, 2));

        return jarak;

    }
//######################################################################
//######################################################################

//######################################################################
/////////////
//method untuk  mencari garis tepi
/////////////
    public static Mat getEdge_2(Mat frame) {
        frame = segment(frame);
        Imgproc.Canny(frame, frame, 0.2, 0.2);
        return frame;
    }

//######################################################################
/////////////
//method untuk  mencari garis tepi
/////////////
    public static Mat getEdge(Mat frame) {
        frame = segment(frame);
//        frame = frame.reshape(1, 1);
        Core.flip(frame, frame, 1);
//        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
        Mat dist = new Mat();
//        batas minimum treshold
        Imgproc.threshold(frame, frame, 100, 255, Imgproc.THRESH_BINARY_INV);
        cleaning(frame);

        Imgproc.resize(frame, frame, new Size(640, 480));
//        Imgproc.Canny(frame, frame, 0.2, 0.2);
        frame = frame.reshape(1, 1);
        return frame;
    }
//######################################################################
//######################################################################

//######################################################################
/////////////
//method untuk memisahkan objek dengan background
//kendala background masih harus bersih dan memiliki warna tidak selaras kulit
/////////////
    public static Mat segmentInvers(Mat frameAsli) {
        try {
            double tres = 0.5;
//        Mat frameUbah = frameAsli.clone();
//            Scalar upperb = new Scalar(64, 223, 255);
//            Scalar lowerb = new Scalar(0, 0, 0);
//            Core.inRange(frameAsli, lowerb, upperb, frameAsli);
//            updateImageView(layarBW, Utils.mat2Image(frameAsli));

            Imgproc.cvtColor(frameAsli, frameAsli, Imgproc.COLOR_BGR2GRAY);
//        Imgproc.accumulateWeighted(frame, frame, 0.5);
            Imgproc.GaussianBlur(frameAsli, frameAsli, new Size(7, 7), 0);
            Mat dist = new Mat();
//        batas minimum treshold
            Imgproc.threshold(frameAsli, frameAsli, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
//            Imgproc.threshold(frameAsli, frameAsli, 100, 255, Imgproc.THRESH_BINARY_INV);

            cleaning(frameAsli);
        } catch (Exception e) {
            System.out.println("segment(Mat frameAsli)");
            System.out.println(e);
            System.out.println("");
        }

        return frameAsli;
    }
//######################################################################
/////////////
//method untuk memisahkan objek dengan background
//kendala background masih harus bersih dan memiliki warna tidak selaras kulit
/////////////

    public static Mat segment(Mat frameAsli) {
        try {
//            double tres = 50.0;
            double tres = .5;
//        Mat frameUbah = frameAsli.clone();
//            Scalar upperb = new Scalar(64, 223, 255);
//            Scalar lowerb = new Scalar(0, 0, 0);
//            Core.inRange(frameAsli, lowerb, upperb, frameAsli);
//            updateImageView(layarBW, Utils.mat2Image(frameAsli));

            Imgproc.cvtColor(frameAsli, frameAsli, Imgproc.COLOR_BGR2GRAY);
//            Imgproc.accumulateWeighted(frameAsli, frameAsli, 10000);
            Imgproc.GaussianBlur(frameAsli, frameAsli, new Size(7, 7), 0);
//            Mat dist = new Mat();
//            batas minimum treshold
            Imgproc.threshold(frameAsli, frameAsli, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
//            Imgproc.adaptiveThreshold(frameAsli, frameAsli, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2);
//            Imgproc.threshold(frameAsli, frameAsli, 100, 255, Imgproc.THRESH_BINARY);
            cleaning(frameAsli);
        } catch (Exception e) {
            System.out.println("segment(Mat frameAsli)");
            System.out.println(e);
            System.out.println("");
        }

        return frameAsli;
    }
//######################################################################
//######################################################################

//######################################################################
/////////////
//method untuk mengambil posisi tangan pada kotak
/////////////
    public static Mat getBox(Mat frame) {
        Rect rectCrop = new Rect(
                new Point(frame.cols() - 5, 10 + 5),
                new Point(frame.cols() / 2 + 5, frame.rows() - (frame.rows() / 3) - 10 - 5)
        );
        frame = frame.submat(rectCrop);
        //# convert the roi to grayscale and blur it
//        Imgproc.accumulateWeighted(frame, frame, 0.5);
        Imgproc.GaussianBlur(frame, frame, new Size(7, 7), 0);
        return frame;
    }

//######################################################################
/////////////
//method untuk mengambil posisi tangan pada kotak
/////////////
    public static Mat getBox(Mat frame, Point p, Point p_) {
        Rect rectCrop = new Rect(
                p,
                p_
        );
        frame = frame.submat(rectCrop);
        //# convert the roi to grayscale and blur it
//        Imgproc.accumulateWeighted(frame, frame, 0.5);
        Imgproc.GaussianBlur(frame, frame, new Size(7, 7), 0);
        return frame;
    }
//######################################################################
/////////////
//method untuk menggambar kotak untuk posisi tangan
/////////////

    public static Mat drawRect(Mat frame) {
        Imgproc.rectangle(frame,
                new Point(frame.cols(), 10),
                new Point(frame.cols() / 2, frame.rows() - (frame.rows() / 3) - 10),
                new Scalar(0, 0, 255),
                3);
        return frame;
    }
//######################################################################
/////////////
//method untuk menggambar kotak untuk posisi tangan
/////////////

    public static Mat drawRect(Mat frame, Point p, Point p_) {
        Imgproc.rectangle(frame,
                p,
                p_,
                new Scalar(0, 0, 255),
                3);
        return frame;
    }
//######################################################################
//######################################################################

//######################################################################
//****************************************************************************//
//****************************************************************************//
//method draw pada frame
//****************************************************************************//
/////////////
//
/////////////
    public static Mat drawContour(List<MatOfPoint> contours, Mat frame) {
        for (int i = 0; i < contours.size(); i++) {
            Imgproc.drawContours(frame, contours, i, new Scalar(0, 255, 0), 3);
        }
        return frame;
    }

//######################################################################
/////////////
//
/////////////
    public static Mat drawContour(List<MatOfPoint> contours, Mat frame, Integer[] Index
    ) {
        for (int i = 0; i < contours.size(); i++) {
            Imgproc.drawContours(frame, contours, i, new Scalar(0, 255, 0), 3);
        }
        return frame;
    }

//######################################################################
/////////////
//
/////////////
    public static Mat drawPointColor(List<MatOfPoint> contours, Mat frame, Integer[] index) {
        for (int i = 0; i < contours.get(0).toArray().length; i++) {
            Scalar s;
            if (i == 0) {
                s = new Scalar(255, 255, 255);
            } else if (i % 3 == 0) {
                s = new Scalar(255, 0, 0);
            } else if (i % 3 == 2) {
                s = new Scalar(0, 255, 0);
            } else {
                s = new Scalar(0, 0, 255);
            }

            if (index[i] != null && index[i] >= 0 && contours.get(0).toArray()[i].x >= 0) {
                Imgproc.circle(frame, contours.get(0).toArray()[i], 10,
                        s, -1);
                System.out.println("DRAW");
//                Imgproc.putText(frame, contours.get(0).toArray()[i].toString(),
//                        contours.get(0).toArray()[i], 2, 0.5, s);

            }
        }
        return frame;
    }

//######################################################################
/////////////
//
/////////////
    public static Mat drawPointColor(List<MatOfPoint> contours, Mat frame) {
        Scalar s;
        for (int i = 0; i < contours.get(0).toArray().length; i++) {

            if (i == 0) {
                s = new Scalar(255, 255, 255);
            } else if (i % 3 == 0) {
                s = new Scalar(255, 0, 0);
            } else if (i % 3 == 2) {
                s = new Scalar(0, 255, 0);
            } else {
                s = new Scalar(0, 0, 255);
            }
            if (contours.get(0).toArray()[i].x >= 0) {
                Imgproc.circle(frame, contours.get(0).toArray()[i], 10,
                        s, -1);
//                Imgproc.putText(frame, contours.get(0).toArray()[i].toString(),
//                        contours.get(0).toArray()[i], 2, 0.5, s);

            }
        }
        return frame;
    }

//######################################################################
/////////////
//
/////////////
    public static Mat drawPointColor(List<MatOfPoint> contours, Mat frame, List<Integer> index) {

        Integer k = -1;
        for (int j = 0; j < index.size(); j++) {

            Scalar s = new Scalar(255, 255, 255);
            if (j == 0) {
                s = new Scalar(255, 255, 255);
            } else if (j % 3 == 0) {
                s = new Scalar(255, 0, 0);
            } else if (j % 3 == 2) {
                s = new Scalar(0, 255, 0);
            } else {
                s = new Scalar(0, 0, 255);
            }
            if (index.get(j) < contours.get(0).toArray().length && index.get(j) >= 0) {
                Imgproc.circle(frame, contours.get(0).toArray()[index.get(j)], 10, s, -1);
            } else {
                k = j;

            }
//                Imgproc.putText(frame, contours.get(0).toArray()[i].toString(),
//                        contours.get(0).toArray()[i], 2, 0.5, s);

        }
        return frame;
    }

//######################################################################
/////////////
//
/////////////
    public static Mat drawJumlahJari(Mat frame, int size) {
        Scalar s;

        s = new Scalar(0, 0, 255);

        Imgproc.putText(frame, "Jumlah Puncak Jari = " + size,
                new Point(10.0, 10.0), 2, 0.5, s);
        System.out.println("Jumlah Puncak Jari = " + size);
//
        return frame;
    }
//######################################################################
//######################################################################

//######################################################################
/////////////
//get list point from dev
//28/08/2018
/////////////
    public static List<MatOfPoint> getContour(Mat frame) {
        Mat hierarchy = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(frame.clone(), contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        if (contours.isEmpty()) {
            Point[] cPoint = null;
        } else {
            MatOfPoint c = contours.get(0);
            Point[] cPoint = c.toArray();
        }

        return contours;
    }

//######################################################################
/////////////
//
/////////////
    public static List<MatOfInt4> getDevectIndexPoint(List<MatOfPoint> contours) {
        List<MatOfInt4> devList = new ArrayList<>();
        List<MatOfInt> hullList = getHullIndexPoint(contours);
//        for (int i = 0; i < hullList.size(); i++) {
//            try {
//                MatOfInt4 dev = new MatOfInt4();
//                Imgproc.convexityDefects(contours.get(i), hullList.get(i), dev);
////                int[] devarr = dev.toArray();
////                for (int j = 0; j < devarr.length; j++) {
////                    System.out.println(devarr[j]);
////                }
//                devList.add(dev);
//            } catch (Exception e) {
//                System.out.println("isi devec");
//                System.out.println(e);
//            }
//        }
        try {

            MatOfInt4 dev = new MatOfInt4();
            MatOfInt hull = hullList.get(0);
            MatOfPoint cont = contours.get(0);
            Imgproc.convexityDefects(cont, hull, dev);
            devList.add(dev);

        } catch (Exception e) {
            System.out.println(
                    "getDevectIndexPoint(List<MatOfPoint> contours");
            System.err.println(e);
            System.out.println("");
        }
        return devList;
    }

//######################################################################
/////////////
//
/////////////
    public static List<MatOfInt> getHullIndexPoint(List<MatOfPoint> contours) {
        List<MatOfInt> hullList = new ArrayList<>();
        for (MatOfPoint contour : contours) {
            MatOfInt hull = new MatOfInt();
            Imgproc.convexHull(contour, hull);
            hullList.add(hull);
        }
        return hullList;
    }
    //method untuk menggambar contour

//######################################################################
/////////////
//
/////////////
    public static List<MatOfPoint> toListMatOfPointHull(List<MatOfPoint> contours, List<MatOfInt> hull) {
        List<MatOfPoint> listPoint = new ArrayList<>();
        for (int j = 0; j < hull.size(); j++) {

            Point[] contourArray = contours.get(j).toArray();
            Point[] hullPoints = new Point[hull.get(j).rows()];
            List<Integer> hullContourIdxList = hull.get(j).toList();
            for (int i = 0; i < hullContourIdxList.size(); i++) {
                hullPoints[i] = contourArray[hullContourIdxList.get(i)];
            }
            listPoint.add(new MatOfPoint(hullPoints));

        }

        return listPoint;
    }

//######################################################################
/////////////
//
/////////////
    public static List<MatOfPoint> toListMatOfPointDevec(List<MatOfPoint> contours, List<MatOfInt4> dev, List<Integer> devContourIdxList) {
        List<MatOfPoint> listPoint = new ArrayList<>();
        try {
            devContourIdxList = dev.get(0).toList();
            Collections.sort(devContourIdxList);
        } catch (Exception e) {
            System.out.println(
                    "toListMatOfPointDevec(List<MatOfPoint> contours, "
                    + "            List<MatOfInt4> dev)");
            System.out.println(e);
            System.out.println("");
        }
//        try {
//            for (int j = 0; j < dev.size(); j++) {
////            System.out.println("iterasi " + j);
//                Point[] contourArray = contours.get(0).toArray();
//                Point[] devPoints = new Point[dev.get(0).rows() * 4];
//                devContourIdxList = dev.get(0).toList();
//                Collections.sort(devContourIdxList);
//
////                dev.get(0);
//                for (int i = 0; i < devContourIdxList.size(); i++) {
//                    if (devContourIdxList.get(i) < contourArray.length /*&& (i == 0 || devContourIdxList.get(i)
//                        - devContourIdxList.get(i
//                                - 1) > 5)*/) {
//                        devPoints[i] = contourArray[devContourIdxList.get(i)];
////                    System.out.println("point " + devPoints[i].toString());
//                    } else {
//                        devPoints[i] = new Point(-1, -1);
//                    }
//                }
//                listPoint.add(new MatOfPoint(devPoints));
//            }
//        } catch (Exception e) {
//            System.out.println(
//                    "toListMatOfPointDevec(List<MatOfPoint> contours, List<MatOfInt4> dev)");
//            System.out.println(e);
//            System.out.println("");
////
//            //error mungkint terjadi pada method ini
////        aa //            //
//        }

        return listPoint;
    }

//######################################################################
/////////////
//
/////////////
    public static List<Point> toListContour(MatOfPoint contours) {
        List<Point> listPoint = new ArrayList<>();
        Point p = new Point();

        for (int i = 0; i < contours.rows(); i++) {

            p = new Point(contours.get(i, 0)[0], contours.get(i, 0)[1]);
//            p.set(contours.get(i, 0));
            listPoint.add(p);
        }
        return listPoint;
    }
//######################################################################
//######################################################################

    public static List<Point> sortPointByX(List<Point> listPoint) {
        Collections.sort(listPoint, new Comparator<Point>() {
            @Override
            public int compare(Point o1, Point o2) {

                return (int) (o1.x - o2.x);
            }
        });
        return listPoint;
    }

    public static List<Point> sortPointByY(List<Point> listPoint) {
        Collections.sort(listPoint, new Comparator<Point>() {
            @Override
            public int compare(Point o1, Point o2) {

                return (int) (o1.y - o2.y);
            }
        });
        return listPoint;
    }
}
