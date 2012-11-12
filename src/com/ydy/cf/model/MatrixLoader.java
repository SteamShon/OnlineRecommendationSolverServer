package com.ydy.cf.model;

import java.io.File;
import java.io.IOException;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.math.Matrix;

import com.ydy.cf.common.VectorUtils;


public class MatrixLoader {
  // since I am lazy now, I will just use dirty static for easyness.
  public static String dataFile;
  public static int solverType = 0;
  private Matrix Y = null;
  private Matrix YtransposeY = null;
  private Matrix Ytranspose = null;
  private DataModel dataModel;
  private FastByIDMap<Float> userWeightSums;
  private FastByIDMap<Float> itemWeightSums;
  private int ratingDimension = -1;
  
  private MatrixLoader() {
    try {
      if (solverType == 0 || solverType == 1) {
        this.Y = VectorUtils.loadDenseMatrixFromFile(dataFile);
        this.YtransposeY = Y.transpose().times(Y);
        // we load item x latent features so numRows of this matrix is input dimension
        ratingDimension = Y.numRows();
      } else if (solverType == 2) {
        this.Y = VectorUtils.loadSparseMatrixFromFile(dataFile);
        this.YtransposeY = Y.transpose().times(Y);
        // we load item x item similarity sparse matrix so either numRows or numCols is input dimension
        ratingDimension = Y.numRows();
      } else if (solverType == 3) {
        // for label propagation, use FastById data structure for speed issue.
        this.dataModel = new FileDataModel(new File(dataFile));
        this.userWeightSums = getWeightSums(true);
        this.itemWeightSums = getWeightSums(false);
        // input rating range 
        ratingDimension = dataModel.getNumItems();
      }
    } catch (IOException e) {
      e.printStackTrace();
    } catch (TasteException e) {
      e.printStackTrace();
    }
  }
  private static class MatrixLoaderHolder {
    public static final MatrixLoader INSTANCE = new MatrixLoader();
  }
  private FastByIDMap<Float> getWeightSums(boolean forUser) throws TasteException {
    LongPrimitiveIterator ids = forUser ? dataModel.getUserIDs() : dataModel.getItemIDs();
    int size = forUser ? dataModel.getNumUsers() : dataModel.getNumItems();
    FastByIDMap<Float> sums = new FastByIDMap<Float>(size);
    while (ids.hasNext()) {
      long id = ids.next();
      float sum = 0f;
      PreferenceArray others = forUser ? dataModel.getPreferencesFromUser(id) : dataModel.getPreferencesForItem(id);
      for (Preference pref : others) {
        sum += pref.getValue();
      }
      sums.put(id,  sum);
    }
    return sums;
  }
  public static MatrixLoader getInstance() {
    return MatrixLoaderHolder.INSTANCE;
  }
  public Matrix getY() {
    return Y;
  }
  public Matrix getYtransposeY() {
    return YtransposeY;
  }
  public Matrix getYtranpose() {
    return this.Ytranspose;
  }
  public DataModel getDataModel() {
    return dataModel;
  }
  public FastByIDMap<Float> getUserWeightSums() {
    return userWeightSums;
  }
  public FastByIDMap<Float> getItemWeightSums() {
    return itemWeightSums;
  }
  public int getRatingDimension() {
    return ratingDimension;
  }
  
}
