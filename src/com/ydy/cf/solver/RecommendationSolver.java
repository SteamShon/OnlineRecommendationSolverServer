package com.ydy.cf.solver;

import java.util.List;

import org.apache.mahout.cf.taste.recommender.RecommendedItem;

public interface RecommendationSolver {
  public List<RecommendedItem> solveAll();
}
