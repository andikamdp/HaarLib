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

    /**
     * method untuk membarsihkan gambar dari titk hitam dan putih dari hasil treshold
     *
     */
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

    /**
     * method untuk memeriksa apakah titik saat ini lebih tinggi dari titik sebelumnya
     * semakin kecil nilai titik pada frame semakin tinggi
     * var:
     * Point titikA : titik saat ini
     * Point titikB : titik sebelumnya
     *
     */
    public static Boolean arahTitikY(Point titikA, Point titikB) {
        if (titikA.y < titikB.y) {
            /* titikA lebih tinggi dari titikB */
            return true;
        } else if (titikA.y == titikB.y) {
            return false;
        } else {
            return false;
        }
    }
//######################################################################

    /**
     * method untuk memeriksa apakah titik saat ini lebih kekanan dari titik sebelumnya
     * semakin kecil nilai titik pada frame semakin jauh kekanan
     * var:
     * Point titikA : titik saat ini
     * Point titikB : titik sebelumnya
     *
     */
    public static Boolean arauTitikX(Point titikA, Point titikB) {
        if (titikA.x > titikB.x) {
            /* titikA lebih ke kanan dari titikB */
            return true;
        } else {
            return false;
        }
    }
//######################################################################

    /**
     * method untuk menghitung jarak antara dua titik
     * hasil pperhitungan akan diprint dalam method
     * var:
     * List<MatOfPoint> contour : list data titik contour objek
     * Point[] point : transfer data dari list ke array Point
     * double jarak : variabel berisi perhitungan jarak dua titik
     *
     */
    public static void hitungJarakTitik(List<MatOfPoint> contour) {
        Point[] point = contour.get(0).toArray();
        for (int j = 0; j < point.length - 1; j++) {
            double jarak = Math.sqrt(
                    Math.pow(point[j].x - point[j + 1].x, 2)
                    + Math.pow(point[j].y - point[j + 1].y, 2));
            System.out.println(point[j].toString() + " : "
                    + point[j + 1] + " " + jarak);
        }
    }
//######################################################################

    /**
     * method untuk menghitung jarak antara dua titik
     * titik yang digunakan sesuai index dalam variabel array index
     * hasil pperhitungan akan diprint dalam method
     * var:
     * List<MatOfPoint> contour : list data titik contour objek
     * Point[] point : transfer data dari list ke array Point
     * double jarak : variabel berisi perhitungan jarak dua titik
     * Integer[] index : variabel array berisi indek titik yang akan di hitung jaraknya
     * int k : variabel menyimpan indek titik sebelum j
     */
    public static void hitungJarakTitik(List<MatOfPoint> contour, Integer[] index) {
        Point[] point = contour.get(0).toArray();
        int k = 0;
        for (int j = 1; j < point.length - 1; j++) {
            if (index[j] != null && index[j] < 0) {
                double jarak = Math.sqrt(Math.pow(point[k].x - point[j].x, 2)
                        + Math.pow(point[k].y - point[j].y, 2));
                System.out.println(point[k].toString() + " : "
                        + point[j] + " " + jarak);
                k = j;
            }
        }
    }
//######################################################################

    /**
     * method untuk menghitung jarak antara dua titik
     * hasil pperhitungan akan dikembalikan oleh method
     * var:
     * Point pointA : berisi titik pertama
     * Point pointb : berisi titik pertama
     * double jarak : variabel berisi perhitungan jarak dua titik
     *
     */
    public static double hitungJarakTitik(Point titikA, Point titikB) {
        double jarak = Math.sqrt(Math.pow(titikA.x - titikB.x, 2)
                + Math.pow(titikA.y - titikB.y, 2));
        return jarak;
    }
//######################################################################

    /**
     * method memperoleh garis tepi pada gambar
     * method akan mengembalikan gambar berisi garis tapi
     * var:
     * Mat frame : berisi gambar awal
     */
    public static Mat getEdge_2(Mat frame) {
        frame = segment(frame);
        Imgproc.Canny(frame, frame, 0.2, 0.2);
        return frame;
    }

//######################################################################
    /**
     * method memperoleh garis tepi pada gambar
     * gambar yang diperoleh akan dilakukan resaping agar diperoleh fitur
     * method akan mengembalikan nilai fitur hasil reshaping
     * var:
     * Mat frame : berisi gambar awal
     *
     */
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

    /**
     * method memisahkan objek dari latar
     * gambar yang diterima akan diubah menjadi BW dengan treshold binaryInverse dan otsu
     * gambar dilakukan bluring dan cleaning
     * var:
     * Mat frameAsli : berisi gambar dengan warna utuh
     * 12/6/18 6:59 AM
     */
    public static Mat segmentInvers(Mat frameAsli) {
        try {
            double tres = 0.5;
//            Scalar upperb = new Scalar(64, 223, 255);
//            Scalar lowerb = new Scalar(0, 0, 0);
//            Core.inRange(frameAsli, lowerb, upperb, frameAsli););

            Imgproc.cvtColor(frameAsli, frameAsli, Imgproc.COLOR_BGR2GRAY);
//        Imgproc.accumulateWeighted(frame, frame, 0.5);
            Imgproc.GaussianBlur(frameAsli, frameAsli, new Size(7, 7), 0);
            Mat dist = new Mat();
            Imgproc.threshold(frameAsli, frameAsli, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
            cleaning(frameAsli);
        } catch (Exception e) {
            System.out.println("segment(Mat frameAsli)");
            System.out.println(e);
            System.out.println("");
        }

        return frameAsli;
    }

    //######################################################################
    /**
     * method memisahkan objek dari latar
     * gambar yang diterima akan diubah menjadi BW dengan treshold binary dan otsu
     * gambar dilakukan bluring dan cleaning
     * var:
     * Mat frameAsli : berisi gambar dengan warna utuh
     * 12/6/18 6:59 AM
     */
    public static Mat segment(Mat frameAsli) {
        try {
//            double tres = 50.0;
            double tres = .5;
//            Scalar upperb = new Scalar(64, 223, 255);
//            Scalar lowerb = new Scalar(0, 0, 0);
//            Core.inRange(frameAsli, lowerb, upperb, frameAsli);
            Imgproc.cvtColor(frameAsli, frameAsli, Imgproc.COLOR_BGR2GRAY);
//            Imgproc.accumulateWeighted(frameAsli, frameAsli, 10000);
            Imgproc.GaussianBlur(frameAsli, frameAsli, new Size(7, 7), 0);
            Imgproc.threshold(frameAsli, frameAsli, 0, 255,
                    Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
            cleaning(frameAsli);
        } catch (Exception e) {
            System.out.println("segment(Mat frameAsli)");
            System.out.println(e);
            System.out.println("");
        }

        return frameAsli;
    }
//######################################################################

    /**
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
        /**
         * convert the roi to grayscale and blur it
         * belum berhasil di jalamkan
         * Imgproc.accumulateWeighted(frame, frame, 0.5);
         */
        Imgproc.GaussianBlur(frame, frame, new Size(7, 7), 0);
        return frame;
    }
//######################################################################

    /**
     * method ekstrak gambar dari frame melalui koordinat yang diberikan
     * method akan mengembalikan gambar setalah dilakukan bluring
     * var:
     * Mat frame : berisi gambar awal dengan dari kamera
     * Point p, p_ : berisi nilai titik ROI
     * Rect rectCrop : berisi koordinat ROI
     * 12/6/18 6:59 AM
     */
    public static Mat getBox(Mat frame, Point p, Point p_) {
        Rect rectCrop = new Rect(p, p_);
        frame = frame.submat(rectCrop);
        //# convert the roi to grayscale and blur it
//        Imgproc.accumulateWeighted(frame, frame, 0.5);
        Imgproc.GaussianBlur(frame, frame, new Size(7, 7), 0);
        return frame;
    }
//######################################################################

    /**
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
//######################################################################

    /**
     * method memisahkan menggambar ROI kotak meraah
     * var:
     * Mat frameAsli : berisi gambar dengan warna utuh
     * 12/6/18 6:59 AM
     */
    public static Mat drawRect(Mat frame, Point p, Point p_) {
        Imgproc.rectangle(frame,
                p,
                p_,
                new Scalar(0, 0, 255),
                3);
        return frame;
    }
//######################################################################

    /**
     * method untuk menggambar garis tepi pada objek
     * var:
     * List<MatOfPoint> contours : berisi list point titik contur
     * Mat frame : berisi gambar dengan warna utuh
     *
     */
    public static Mat drawContour(List<MatOfPoint> contours, Mat frame) {
        for (int i = 0; i < contours.size(); i++) {
            Imgproc.drawContours(frame, contours, i, new Scalar(0, 255, 0), 3);
        }
        return frame;
    }
//######################################################################

    /**
     * method untuk menggambar titik
     * var:
     * List<MatOfPoint> contours : berisi list point titik contur
     * Mat frame : berisi gambar dengan warna utuh
     * Integer[] Index : index titik
     *
     */
    public static Mat drawContour(List<MatOfPoint> contours, Mat frame, Integer[] Index
    ) {
        for (int i = 0; i < contours.size(); i++) {
            Imgproc.drawContours(frame, contours, i, new Scalar(0, 255, 0), 3);
        }
        return frame;
    }
//######################################################################

    /**
     * method untuk menggambar titik
     * var:
     * List<MatOfPoint> contours : berisi list point titik contur
     * Mat frame : berisi gambar dengan warna utuh
     * Integer[] Index : index titik
     *
     */
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
//                Imgproc.putText(frame, contours.get(0).toArray()[i].toString(),
//                        contours.get(0).toArray()[i], 2, 0.5, s);

            }
        }
        return frame;
    }
//######################################################################

    /**
     * method untuk menggambar titik
     * var:
     * List<MatOfPoint> contours : berisi list point titik contur
     * Mat frame : berisi gambar dengan warna utuh
     *
     */
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

    /**
     * method untuk menuliskan jumlah jari pada gambar
     * var:
     * List<MatOfPoint> contours : berisi list point titik contur
     * Mat frame : berisi gambar dengan warna utuh
     * int size : ukuran font huruf
     *
     */
    public static Mat drawJumlahJari(Mat frame, int size) {
        Scalar s;
        s = new Scalar(0, 0, 255);
        Imgproc.putText(frame, "Jumlah Puncak Jari = " + size,
                new Point(10.0, 10.0), 2, 0.5, s);
        return frame;
    }
//######################################################################

    /**
     * method untuk memperoleh nilai titik tepi pada gambar
     * gambar yang diterima telah di treshold
     * var:
     * Mat frame : berisi gambar dengan warna biner
     * Mat hierarchy :
     *
     */
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

    /**
     * method untuk mengrutkan nilai titik(point) berdasarkan x
     * var:
     * List<Point> listPoint : berisi nilai titik tepi
     * Mat hierarchy :
     *
     */
    public static List<Point> sortPointByX(List<Point> listPoint) {
        Collections.sort(listPoint, new Comparator<Point>() {
            @Override
            public int compare(Point o1, Point o2) {
                return (int) (o1.x - o2.x);
            }
        });
        return listPoint;
    }
//######################################################################

    /**
     * method untuk mengrutkan nilai titik(point) berdasarkan Y
     * var:
     * List<Point> listPoint : berisi nilai titik tepi
     * Mat hierarchy :
     *
     */
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
