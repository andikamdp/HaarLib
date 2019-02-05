/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package src.controller;

import java.io.File;
import java.net.URL;
import java.util.ResourceBundle;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.AnchorPane;
import javafx.stage.DirectoryChooser;

/**
 * FXML Controller class
 *
 * @author Andika Mulyawan
 */
public class RenameDataController implements Initializable {

    @FXML
    private Button btnBrowseFile;
    @FXML
    private TextField txtBoxFileLocation;
    @FXML
    private Label lblFileLength;
    @FXML
    private TextField txtBoxFileName;
    @FXML
    private Button btnRenameFile;
    @FXML
    private AnchorPane apRenameWindow;
    //
    private File path;
    private MainAppController mainAppController;

    /**
     * Initializes the controller class.
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
    }

    @FXML
    private void browseFileOnClick(ActionEvent event) {
        DirectoryChooser brows = new DirectoryChooser();
        brows.setTitle("Buka Folder Data Training");
        path = brows.showDialog(apRenameWindow.getScene().getWindow());
        if (path != null) {
            lblFileLength.setText(String.valueOf(path.listFiles().length));
            txtBoxFileLocation.setText(path.getAbsolutePath());
        }
    }

    @FXML
    private void renameFileOnClick(ActionEvent event) {
        File[] paths = path.listFiles();
        int i = 0;
        for (File file : paths) {
            boolean p = file.renameTo(new File(file.getParent() + "\\" + txtBoxFileName.getText() + "_" + i + ".jpg"));
            i++;
        }
    }

    void setMainController(MainAppController aThis) {
        mainAppController = aThis;
    }

}
