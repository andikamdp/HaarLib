/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package src.controller;

import java.net.URL;

import java.util.ResourceBundle;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.scene.layout.AnchorPane;
import src.controller.HaarCobaController;
import java.io.IOException;
import java.util.logging.Logger;
import javafx.scene.Scene;

import java.util.logging.Level;
import src.MainApp;

/**
 * FXML Controller class
 *
 * @author Andika Mulyawan
 */
public class MainAppController implements Initializable {

    @FXML
    private Button btnSvmTrain;
    @FXML
    private Button btnSvmPredict;
    @FXML
    private AnchorPane mainAppView;
    @FXML
    private Button btnSetImage;
    @FXML
    private Button btnRenameData;

    /**
     * Initializes the controller class.
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
    }

    private Stage stage;

    /**
     * onClick action membuka window baru berisi gui training SVM
     *
     */
    @FXML
    private void svmTrainViewOnClick(ActionEvent event) {
        try {
            FXMLLoader loader = new FXMLLoader();
            loader.setLocation(MainApp.class.getResource("view/SVMTrain.fxml"));
            AnchorPane root = loader.load();
            SVMTrainController controller = loader.getController();
            controller.setMainController(this);
            Scene scene = new Scene(root);
            stage = new Stage();
            stage.initOwner(mainAppView.getScene().getWindow());
            stage.initModality(Modality.WINDOW_MODAL);
            stage.setScene(scene);
            stage.show();
        } catch (IOException iOException) {
            Logger.getLogger(MainAppController.class.getName()).
                    log(Level.SEVERE, null, iOException);
        }
    }

    /**
     * onClick action membuka window baru berisi live capture image
     *
     */
    @FXML
    private void handViewOnClick(ActionEvent event) {
        try {
            FXMLLoader loader = new FXMLLoader();
            loader.setLocation(MainApp.class.getResource("view/HandView.fxml"));
            AnchorPane root = loader.load();
            HandViewController controller = loader.getController();
            controller.setMainController(this);
            Scene scene = new Scene(root);
            stage = new Stage();
            stage.initOwner(mainAppView.getScene().getWindow());
            stage.initModality(Modality.WINDOW_MODAL);
            stage.setScene(scene);
            stage.show();
        } catch (IOException iOException) {
            Logger.getLogger(MainAppController.class.getName()).
                    log(Level.SEVERE, null, iOException);
        }
    }

    /**
     * onClick action membuka window baru berisi predict pergambar
     */
    @FXML
    private void buildImageDatasetViewOnClick(ActionEvent event) {
        try {
            FXMLLoader loader = new FXMLLoader();
            loader.setLocation(MainApp.class.getResource("view/BuildImageDataset.fxml"));
            AnchorPane root = loader.load();
            BuildImageDatasetController controller = loader.getController();
            controller.setMainController(this);
            Scene scene = new Scene(root);
            stage = new Stage();
            stage.initOwner(mainAppView.getScene().getWindow());
            stage.initModality(Modality.WINDOW_MODAL);
            stage.setScene(scene);
            stage.show();
        } catch (IOException iOException) {
            Logger.getLogger(MainAppController.class.getName()).
                    log(Level.SEVERE, null, iOException);
        }
    }

    /**
     * onClick action membuka window baru untuk melakukan rename file
     */
    @FXML
    private void renameFileViewOnClick(ActionEvent event) {
        try {
            FXMLLoader loader = new FXMLLoader();
            loader.setLocation(MainApp.class.getResource("view/RenameData.fxml"));
            AnchorPane root = loader.load();
            RenameDataController controller = loader.getController();
            controller.setMainController(this);
            Scene scene = new Scene(root);
            stage = new Stage();
            stage.initOwner(mainAppView.getScene().getWindow());
            stage.initModality(Modality.WINDOW_MODAL);
            stage.setScene(scene);
            stage.show();
        } catch (IOException iOException) {
            Logger.getLogger(MainAppController.class.getName()).
                    log(Level.SEVERE, null, iOException);
        }
    }
}
