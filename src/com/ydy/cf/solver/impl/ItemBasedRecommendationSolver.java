package com.ydy.cf.solver.impl;

import java.util.Iterator;
import java.util.List;

import org.apache.mahout.cf.taste.common.TopK;
import org.apache.mahout.cf.taste.impl.recommender.GenericRecommendedItem;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;

import com.ydy.cf.common.VectorUtils;
import com.ydy.cf.model.MatrixLoader;
/*
 * assumes MatrixLoader load item-item similarity matrix into memory
 * 
 */
public class ItemBasedRecommendationSolver extends AbstractRecommendationSolver {
  public ItemBasedRecommendationSolver (String userId, Vector userRatings, MatrixLoader loader, int numRecommendations) {
    super(userId, userRatings, loader, numRecommendations);
    System.out.println(ItemBasedRecommendationSolver.class.getName() + "\t" + userId + "\t" + userRatings);
  }
  
  public List<RecommendedItem> solveAll() {
    final TopK<RecommendedItem> recs = new TopK<RecommendedItem>(numRecommendations, VectorUtils.BY_PREFERENCE_VALUE);
    final Vector ratings = this.userRatings;
    Iterator<MatrixSlice> rows = Y.iterator();
    while (rows.hasNext()) {
      MatrixSlice row = rows.next();
      double score = ratings.dot(row.vector());
      recs.offer(new GenericRecommendedItem(row.index(), (float) (score)));
    }
    return recs.retrieve();
  }
}
