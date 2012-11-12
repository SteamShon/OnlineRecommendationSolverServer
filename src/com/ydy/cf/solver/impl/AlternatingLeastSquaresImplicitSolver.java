/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ydy.cf.solver.impl;

import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.mahout.cf.taste.common.TopK;
import org.apache.mahout.cf.taste.impl.recommender.GenericRecommendedItem;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.QRDecomposition;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

import com.google.common.base.Preconditions;
import com.ydy.cf.common.VectorUtils;
import com.ydy.cf.model.MatrixLoader;

/** see <a href="http://research.yahoo.com/pub/2433">Collaborative Filtering for Implicit Feedback Datasets</a> 
 * pre-loaded:
 *  1) userRatings
 *  2) matrix M
 *  2) matrix M'M
 * 
 * */

public class AlternatingLeastSquaresImplicitSolver extends AbstractRecommendationSolver {
  private final int numFeatures;
  private final double alpha;
  private final double lambda;
  private final Matrix YtransposeY;
  
  public AlternatingLeastSquaresImplicitSolver(
      String userId, Vector userRatings, double lambda, double alpha, 
      MatrixLoader loader, int numRecommendations) {
    super(userId, userRatings, loader, numRecommendations);
    this.lambda = lambda;
    this.alpha = alpha;
    this.YtransposeY = loader.getYtransposeY();
    this.numFeatures = this.Y.columnSize();
    System.out.println(AlternatingLeastSquaresImplicitSolver.class.getName() + "\tuserRatings: " + userId + "\t" + this.userRatings);
  }
  public List<RecommendedItem> solveAll() {
    Vector userFeatures = solve(this.userRatings);
    return VectorUtils.buildRecommends(this.Y, this.userRatings, userFeatures, 100).retrieve();
  }
  public Vector solve(Vector ratings) {
    Matrix A = YtransposeY.plus(YtransponseCuMinusIYPlusLambdaI(ratings));
    Matrix y = YtransponseCuPu(ratings);
    Vector solved = solve(A, y);
    return solved;
  }
  
  private static Vector solve(Matrix A, Matrix y) {
    return new QRDecomposition(A).solve(y).viewColumn(0);
  }
  
  protected double confidence(double rating) {
    return 1 + alpha * rating;
  }
  
  /** Y' (Cu - I) Y + λ I */
  private Matrix YtransponseCuMinusIYPlusLambdaI(Vector userRatings) {
    Preconditions.checkArgument(userRatings.isSequentialAccess(), "need sequential access to ratings!");

    /* (Cu -I) Y */
    OpenIntObjectHashMap<Vector> CuMinusIY = new OpenIntObjectHashMap<Vector>();
    Iterator<Vector.Element> ratings = userRatings.iterateNonZero();
    while (ratings.hasNext()) {
      Vector.Element e = ratings.next();
      Vector curYRow = Y.viewRow(e.index());
      CuMinusIY.put(e.index(), curYRow.times(confidence(e.get()) - 1));
    }

    Matrix YtransponseCuMinusIY = new DenseMatrix(numFeatures, numFeatures);

    /* Y' (Cu -I) Y by outer products */
    ratings = userRatings.iterateNonZero();
    while (ratings.hasNext()) {
      Vector.Element e = ratings.next();
      for (Vector.Element feature : Y.viewRow(e.index())) {
        Vector partial = CuMinusIY.get(e.index()).times(feature.get());
        YtransponseCuMinusIY.viewRow(feature.index()).assign(partial, Functions.PLUS);
      }
    }

    /* Y' (Cu - I) Y + λ I  add lambda on the diagonal */
    for (int feature = 0; feature < numFeatures; feature++) {
      YtransponseCuMinusIY.setQuick(feature, feature, YtransponseCuMinusIY.getQuick(feature, feature) + lambda);
    }

    return YtransponseCuMinusIY;
  }

  /** Y' Cu p(u) */
  private Matrix YtransponseCuPu(Vector userRatings) {
    Preconditions.checkArgument(userRatings.isSequentialAccess(), "need sequential access to ratings!");

    Vector YtransponseCuPu = new DenseVector(numFeatures);

    Iterator<Vector.Element> ratings = userRatings.iterateNonZero();
    while (ratings.hasNext()) {
      Vector.Element e = ratings.next();
      Vector curYRow = Y.viewRow(e.index());
      YtransponseCuPu.assign(curYRow.times(confidence(e.get())), Functions.PLUS);
    }

    return columnVectorAsMatrix(YtransponseCuPu);
  }

  private Matrix columnVectorAsMatrix(Vector v) {
    Matrix matrix = new DenseMatrix(numFeatures, 1);
    for (Vector.Element e : v) {
      matrix.setQuick(e.index(), 0, e.get());
    }
    return matrix;
  }
  
  public TopK<RecommendedItem> buildRecommends(Vector userRatings, Vector userFeatures, int topK) {
    final Map<Integer, Boolean> alreadyRated = VectorUtils.keys(userRatings);
    final TopK<RecommendedItem> topKItems = new TopK<RecommendedItem>(topK, VectorUtils.BY_PREFERENCE_VALUE);
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
  
}
