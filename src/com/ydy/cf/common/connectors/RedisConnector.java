package com.ydy.cf.common.connectors;

import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;

import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;

import com.google.common.collect.Maps;
import com.ydy.cf.common.VectorUtils;

public class RedisConnector {
  private JedisPool pool;
  private Jedis jedis;
  
  public RedisConnector(String connection) {
    this.pool = new JedisPool(new JedisPoolConfig(), connection);
    this.jedis = this.pool.getResource();
  }
  public JedisPool getPool() {
    return pool;
  }
  public void setPool(JedisPool pool) {
    this.pool = pool;
  }
  public Jedis getJedis() {
    return jedis;
  }
  public void setJedis(Jedis jedis) {
    this.jedis = jedis;
  }
  public Map<String, String> fetchAll(int dbIndex) {
    Map<String, String> m = Maps.newHashMap();
    try {
      jedis.select(dbIndex);
      Set<String> keys = jedis.keys("*");
      for (String key : keys) {
        m.put(key, jedis.get(key));
      }
    } finally {
      
    }
    return m;
  }
  public Matrix fetchAllAsMatrix(int dbIndex) {
    Matrix matrix = null;
    int maxRowId = -1;
    try {
      jedis.select(dbIndex);
      Set<String> keys = jedis.keys("*");
      for (String key : keys) {
        int rowId = Integer.parseInt(key);
        maxRowId = Math.max(maxRowId, rowId);
      }
      int cnt = 0;
      for (String key : keys) {
        if (cnt++ % 1000 == 0) {
          System.out.println("<< " + cnt + " rows loaded");
        }
        if (matrix == null) {
          matrix = new DenseMatrix(maxRowId+1, VectorUtils.parseVector(jedis.get(key)).size());
        }
        int rowId = Integer.parseInt(key);
        matrix.assignRow(rowId, VectorUtils.parseVector(jedis.get(key)));
      }
    } finally {
      
    }
    return matrix;
  }
  public List<String> fetchList(int dbIndex, String key) {
    jedis.select(dbIndex);
    return jedis.lrange(key, 0, -1);
  }
  public String fetch(int dbIndex, String key) {
    String ret = null;
    try {
      jedis.select(dbIndex);
      ret = jedis.get(key);
    } finally {
      
    }
    return ret;
  }
}
