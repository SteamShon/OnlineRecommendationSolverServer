package com.ydy.cf.common;

import java.io.File;
import java.io.IOException;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.mahout.cf.taste.common.TopK;
import org.apache.mahout.cf.taste.impl.recommender.GenericRecommendedItem;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.FileLineIterator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Maps;
import com.google.common.primitives.Floats;
/**
 * utils for parsing matrix files.
 * @author ydy
 *
 */
public class VectorUtils {
  public static final String VECTOR_DELIMITER = "\t";
  private static final Logger log = LoggerFactory.getLogger(VectorUtils.class);
  
  public static Map<Integer, Boolean> keys(Vector v) {
    Map<Integer, Boolean> keys = Maps.newHashMap();
    Iterator<Vector.Element> iter = v.iterateNonZero();
    while (iter.hasNext()) {
      Vector.Element e = iter.next();
      keys.put(e.index(), true);
    }
    return keys;
  }
  public static Matrix parseMatrix(Map<String, String> keyValues) {
    Matrix matrix = null;
    int maxRowId = -1;
    for (Entry<String, String> keyValue : keyValues.entrySet()) {
      int rowId = Integer.parseInt(keyValue.getKey());
      maxRowId = Math.max(maxRowId, rowId);
    }
    
    for (Entry<String, String> keyValue : keyValues.entrySet()) {
      if (matrix == null) {
        matrix = new DenseMatrix(maxRowId+1, parseVector(keyValue.getValue()).size());
      }
      int rowId = Integer.parseInt(keyValue.getKey());
      matrix.assignRow(rowId, parseVector(keyValue.getValue()));
    }
    return matrix;
  }
  
  public static Vector parseVector(String s) {
    return parseVector(s, -1);
  }
  public static Pair<Integer, Vector> parseMatrixRow(String s, int numCols) {
    String[] tokens = s.trim().split(VECTOR_DELIMITER);
    if (tokens.length != 2){
      return null;
    }
    return new Pair<Integer, Vector>(Integer.parseInt(tokens[0]), parseVector(tokens[1], numCols));
  }
  public static Pair<Integer, Vector> parseMatrixRow(String s) {
    String[] tokens = s.trim().split(VECTOR_DELIMITER);
    if (tokens.length != 2){
      return null;
    }
    return new Pair<Integer, Vector>(Integer.parseInt(tokens[0]), parseVector(tokens[1]));
  }
  
  public static Vector parseVector(String s, int dim) {
    return parseVector(s, dim, false);
  }
  
  public static Vector parseVector(String s, int dim, boolean isRandomAccess) {
    String inner = s.replaceAll("[{}()]", "");
    String[] tokens = inner.split(",");
    int card = tokens.length;
    if (dim > 0) {
      card = dim;
    }
    Vector v;
    if (isRandomAccess) {
      v = new RandomAccessSparseVector(dim);
    } else {
      v = new SequentialAccessSparseVector(card);
    }
    for (String token : tokens) {
      Pair<Integer, Double> pair = parseElement(token);
      if (pair != null) v.set(pair.getFirst(), pair.getSecond());
    }
    return v;
  }
  
 
  public static Vector parseVector(List<String> strs) {
    Vector v = new SequentialAccessSparseVector(strs.size());
    for (String s : strs) {
      Pair<Integer, Double> pair = parseElement(s);
      v.set(pair.getFirst(), pair.getSecond());
    }
    return v;
  }
  public static Pair<Integer, Double> parseElement(String s) {
    String[] token = s.split(":");
    if (token.length != 2) {
      return null;
    }
    return new Pair<Integer, Double>(Integer.parseInt(token[0]), Double.parseDouble(token[1]));
  }
  
  
  public static Vector randomSparseVector(int dim) {
    Vector v = new RandomAccessSparseVector(dim);
    Random random = new Random();
    for (int i = 0; i < dim; i++) {
      v.set(i, random.nextDouble());
    }
    return v;
  }
  public static Vector randomDenseVector(int dim) {
    Vector v = new DenseVector(dim);
    Random random = new Random();
    for (int i = 0; i < dim; i++) {
      v.set(i, random.nextDouble());
    }
    return v;
  }
  public static Vector randomSequentialVector(int dim) {
    Vector v = new SequentialAccessSparseVector(dim);
    Random random = new Random();
    for (int i = 0; i < dim; i++) {
      v.set(i, random.nextDouble());
    }
    return v;
  }
  
  public static final Comparator<RecommendedItem> BY_PREFERENCE_VALUE =
      new Comparator<RecommendedItem>() {
    @Override
    public int compare(RecommendedItem one, RecommendedItem two) {
      return Floats.compare(one.getValue(), two.getValue());
    }
  };
  
  public static TopK<RecommendedItem> buildRecommends(Matrix Y, Vector userRatings, Vector userFeatures, int topK) {
    final Map<Integer, Boolean> alreadyRated = VectorUtils.keys(userRatings);
    final TopK<RecommendedItem> topKItems = new TopK<RecommendedItem>(topK, BY_PREFERENCE_VALUE);
    Iterator<MatrixSlice> rows = Y.iterator();
    while (rows.hasNext()) {
      MatrixSlice row = rows.next();
      int itemId = row.index();
      Vector itemFeatures = row.vector();
      if (!alreadyRated.containsKey(itemId)) {
        double predictedRating = userFeatures.dot(itemFeatures);
        topKItems.offer(new GenericRecommendedItem(itemId, (float) predictedRating));
      }
    }
    return topKItems;
  }
  
  
  public static Matrix loadDenseMatrixFromFile(File file) throws IOException {
    int numRows = -1, numCols = -1;
    FileLineIterator lines = new FileLineIterator(file);
    while (lines.hasNext()) {
      Pair<Integer, Vector> row = parseMatrixRow(lines.next());
      numRows = Math.max(numRows,  row.getFirst());
      numCols = row.getSecond().size();
    }
    lines.close();
    // +1 is buggy. since index start from 0, need to +1 for dimension for now
    // change this later.
    Matrix matrix = new DenseMatrix(numRows+1, numCols);
    lines = new FileLineIterator(file);
    while (lines.hasNext()) {
      Pair<Integer, Vector> row = parseMatrixRow(lines.next());
      matrix.assignRow(row.getFirst(), row.getSecond());
    }
    return matrix;
  }
  public static Matrix loadDenseMatrixFromFile(String filename) throws IOException {
    return loadDenseMatrixFromFile(new File(filename));
  }
  public static Pair<Integer, Integer> getMaxRowColIds(String filename, boolean matrixFormat) throws IOException {
    log.info("getting max Ids on row, col: " + filename);
    int maxRowId = 0, maxColId = 0;
    FileLineIterator lines = new FileLineIterator(new File(filename));
    while (lines.hasNext()) {
      String line = lines.next();
      if (matrixFormat) {
        String[] tokens = line.split(VECTOR_DELIMITER);
        int rowId = Integer.parseInt(tokens[0]);
        maxRowId = Math.max(maxRowId, rowId);
        if (tokens.length == 2) {
          for (String element : tokens[1].replaceAll("[{}()]", "").split(",")) {
            Pair<Integer, Double> pair = parseElement(element);
            maxColId = Math.max(maxColId, pair.getFirst());
          }
        }
      } else {
        String[] tokens = line.split(",");
        if (tokens.length != 3) continue;
        int rowId = Integer.parseInt(tokens[0]);
        int colId = Integer.parseInt(tokens[1]);
        maxRowId = Math.max(maxRowId, rowId);
        maxColId = Math.max(maxColId, colId);
      }
    }
    lines.close();
    return new Pair<Integer, Integer>(maxRowId, maxColId);
  }
  
  public static Matrix loadSparseMatrixFromFile(String filename) throws IOException {
    Pair<Integer, Integer> maxIds = getMaxRowColIds(filename, true);
    log.info("load sparse matrix from file: " + filename);
    Matrix matrix = new SparseMatrix(maxIds.getFirst() + 1, maxIds.getSecond() + 1);
    FileLineIterator lines = new FileLineIterator(new File(filename));
    while (lines.hasNext()) {
      String line = lines.next();
      String[] tokens = line.split(VECTOR_DELIMITER);
      int rowId = Integer.parseInt(tokens[0]);
      if (tokens.length == 2) {
        Vector cols = new RandomAccessSparseVector(maxIds.getSecond() + 1);
        for (String element : tokens[1].replaceAll("[{}()]", "").split(",")) {
          Pair<Integer, Double> pair = parseElement(element);
          cols.setQuick(pair.getFirst(), pair.getSecond());
        }
        matrix.assignRow(rowId, cols);
      }
    }
    return matrix;
  }
  public static Matrix loadSparseMatrixFromFlatFile(String filename, boolean transpose) throws IOException {
    Pair<Integer, Integer> maxIds = getMaxRowColIds(filename, false);
    int numRows = transpose ? maxIds.getSecond() + 1 : maxIds.getFirst() + 1;
    int numCols = transpose ? maxIds.getFirst() + 1 : maxIds.getSecond() + 1;
    log.info("load sparse matrix from flat file: " + filename);
    Matrix matrix = new SparseMatrix(numRows, numCols);
    FileLineIterator lines = new FileLineIterator(new File(filename));
    while (lines.hasNext()) {
      String line = lines.next();
      String[] tokens = line.split(",");
      if (tokens.length == 3) {
        int rowId = transpose ? Integer.parseInt(tokens[1]) : Integer.parseInt(tokens[0]);
        int colId = transpose ? Integer.parseInt(tokens[0]) : Integer.parseInt(tokens[1]);
        double value = Double.parseDouble(tokens[2]);
        matrix.setQuick(rowId, colId, value);
      }
    }
    return matrix;
  }
}
