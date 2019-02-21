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
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.AnchorPane;

/**
 * FXML Controller class
 *
 * @author Andika Mulyawan
 */
public class BuildImageDatasetController implements Initializable {

    @FXML
    private AnchorPane apGetImage;
    @FXML
    private Button btnStartCamera;
    @FXML
    private Button btnBrowsSaveFile;
    @FXML
    private TextField txtSaveFileLocation;
    @FXML
    private Button btnCreateFolder;
    @FXML
    private TextField txtFolderName;
    @FXML
    private TextField txtIndex;
    @FXML
    private TextField txtIndexLabel;
    @FXML
    private ImageView layarMain;

    /**
     * Initializes the controller class.
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
    }

    @FXML
    private void startCameraOnClick(ActionEvent event) {
    }

    @FXML
    private void browsSaveFileOnClick(ActionEvent event) {
    }

    @FXML
    private void createFolderOnClick(ActionEvent event) {
    }

    @FXML
    private void getFramePointOnClick(MouseEvent event) {
    }

    void setMainController(MainAppController aThis) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
