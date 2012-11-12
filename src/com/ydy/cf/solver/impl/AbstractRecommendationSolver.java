package com.ydy.cf.solver.impl;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import com.ydy.cf.model.MatrixLoader;
import com.ydy.cf.solver.RecommendationSolver;
/**
 * 
 * @author ydy
 *
 */
public abstract class AbstractRecommendationSolver implements RecommendationSolver {
  protected final String userId;
  protected final Matrix Y;
  protected final Vector userRatings;
  protected final int numRecommendations;
  
  public AbstractRecommendationSolver(String userId, Vector userRatings, MatrixLoader loader, int numRecommendations) {
    this.userId = userId;
    this.userRatings = userRatings;
    this.Y = loader.getY();
    this.numRecommendations = numRecommendations;
  }
}
