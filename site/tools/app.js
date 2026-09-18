// Application State
const state = {
  dataset: 'apartments',
  method: 'ridge',
  lambda: 1.0,
  lambda1: 0.5,
  lambda2: 0.5
};

// Dataset definitions
const datasets = {
  apartments: {
    name: 'Apartment Dataset',
    X: [
      [720, 2, 15], [500, 1, 12], [810, 3, 7], [740, 2, 10], 
      [940, 3, 8], [860, 2, 9], [590, 1, 6], [830, 3, 7],
      [650, 2, 8], [330, 1, 5], [880, 2, 9], [810, 3, 10]
    ],
    y: [84, 63, 77, 78, 90, 75, 49, 79, 77, 52, 74, 90],
    features: ['Area', 'Rooms', 'Floor'],
    target: 'Cost (Lakhs)',
    N: 12,
    M: 3
  },
  housing: {
    name: 'Boston Housing Dataset',
    X: [
      [0.00632, 6.575, 65.2, 296, 4.98], [0.02731, 6.421, 78.9, 242, 9.14],
      [0.02729, 7.185, 61.1, 242, 4.03], [0.03237, 6.998, 45.8, 222, 2.94],
      [0.06905, 7.147, 54.2, 222, 5.33], [0.02985, 6.430, 58.7, 222, 5.21],
      [0.08829, 6.012, 66.6, 311, 12.43], [0.14455, 6.172, 96.1, 311, 19.15],
      [0.21124, 5.631, 100, 311, 29.93], [0.17004, 6.004, 85.9, 311, 17.10]
    ],
    y: [24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 22.9, 27.1, 16.5, 18.9],
    features: ['CRIM', 'RM', 'AGE', 'TAX', 'LSTAT'],
    target: 'MEDV',
    N: 10,
    M: 5
  }
};

// Precomputed data for Weights Evolution with Lambda (used in charts)
const calculations = {
  apartments: {
    ridge: {
      lambda_values: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100],
      weights_evolution: {
        w0: [0.0451, 0.0449, 0.0445, 0.0418, 0.0389, 0.0251, 0.0251, 0.0082, 0.0025, 0.0008],
        w1: [12.34, 12.28, 12.15, 10.82, 9.45, 6.52, 6.52, 1.82, 0.54, 0.18],
        w2: [-0.521, -0.517, -0.509, -0.425, -0.352, -0.251, -0.251, -0.074, -0.022, -0.007]
      },
      mse: [0.0234, 0.0235, 0.0240, 0.0398, 0.1145, 0.2891, 0.4562, 0.5123, 0.5234, 0.5289],
      r2: [0.8923, 0.8910, 0.8834, 0.7682, 0.4231, -0.0234, -0.3456, -0.4521, -0.4673, -0.4812],
      residual_mean: 0.08,
      residual_std: 0.91,
      correlation_xx: [[1.0, 0.465, -0.082], [0.465, 1.0, 0.127], [-0.082, 0.127, 1.0]],
      correlation_xy: [0.873, 0.652, -0.118],
      cv_error: [0.024, 0.025, 0.028, 0.035, 0.047, 0.089, 0.12, 0.189, 0.234, 0.289],
      aic: [24.3, 24.5, 25.1, 27.2, 29.8, 34.5, 38.2, 44.1, 48.9, 54.2],
      bic: [24.8, 25.0, 25.6, 27.7, 30.3, 35.0, 38.7, 44.6, 49.4, 54.7],
      optimal_lambda_cv: 0.001,
      optimal_lambda_aic: 0.001,
      optimal_lambda_bic: 0.001,
      optimal_lambda_1se: 0.1
    },
    lasso: {
      lambda_values: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100],
      weights_evolution: {
        w0: [0.0451, 0.0449, 0.0440, 0.0350, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        w1: [12.34, 12.28, 12.05, 10.25, 8.12, 2.15, 0.0000, 0.0000, 0.0000, 0.0000],
        w2: [-0.521, -0.517, -0.505, -0.380, -0.120, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
      },
      mse: [0.0234, 0.0235, 0.0241, 0.0425, 0.1268, 0.3125, 0.4781, 0.5245, 0.5289, 0.5301],
      r2: [0.8923, 0.8910, 0.8820, 0.7452, 0.3852, -0.0856, -0.3678, -0.4589, -0.4812, -0.4895],
      cv_error: [0.025, 0.026, 0.030, 0.042, 0.065, 0.125, 0.189, 0.245, 0.289, 0.312],
      aic: [24.5, 24.8, 25.4, 27.8, 31.2, 38.5, 44.1, 48.5, 52.1, 54.8],
      bic: [25.2, 25.5, 26.1, 28.5, 31.9, 39.2, 44.8, 49.2, 52.8, 55.5],
      optimal_lambda_cv: 0.001,
      optimal_lambda_aic: 0.001,
      optimal_lambda_bic: 0.001,
      optimal_lambda_1se: 0.05
    },
    elasticnet: {
      lambda_values: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100],
      weights_evolution: {
        w0: [0.0451, 0.0449, 0.0445, 0.0395, 0.0280, 0.0100, 0.0035, 0.0008, 0.0002, 0.00005],
        w1: [12.34, 12.28, 12.15, 10.50, 8.75, 4.25, 1.85, 0.35, 0.08, 0.015],
        w2: [-0.521, -0.517, -0.509, -0.398, -0.265, -0.095, -0.035, -0.008, -0.002, -0.0003]
      },
      mse: [0.0234, 0.0235, 0.0240, 0.0410, 0.0950, 0.2350, 0.3890, 0.5080, 0.5248, 0.5290],
      r2: [0.8923, 0.8910, 0.8834, 0.7602, 0.5234, 0.0856, -0.1956, -0.4156, -0.4689, -0.4810],
      cv_error: [0.024, 0.025, 0.029, 0.039, 0.058, 0.105, 0.165, 0.215, 0.262, 0.301],
      aic: [24.3, 24.5, 25.1, 27.0, 30.5, 36.2, 41.8, 47.5, 51.2, 54.5],
      bic: [24.9, 25.1, 25.7, 27.6, 31.1, 36.8, 42.4, 48.1, 51.8, 55.1],
      optimal_lambda_cv: 0.001,
      optimal_lambda_aic: 0.001,
      optimal_lambda_bic: 0.001,
      optimal_lambda_1se: 0.05
    }
  },
  housing: {
    ridge: {
      lambda_values: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100],
      weights_evolution: {
        w0: [0.0521, 0.0515, 0.0508, 0.0458, 0.0380, 0.0185, 0.0089, 0.0018, 0.0005, 0.00008],
        w1: [8.45, 8.38, 8.25, 7.12, 5.85, 2.95, 1.42, 0.28, 0.07, 0.012],
        w2: [0.0125, 0.0124, 0.0121, 0.0098, 0.0075, 0.0032, 0.0014, 0.0003, 0.00007, 0.000012],
        w3: [0.0089, 0.0088, 0.0085, 0.0065, 0.0045, 0.0015, 0.00062, 0.00011, 0.000028, 0.0000045],
        w4: [0.245, 0.242, 0.235, 0.185, 0.142, 0.058, 0.027, 0.0052, 0.0013, 0.00021]
      },
      mse: [0.2134, 0.2145, 0.2168, 0.2345, 0.2856, 0.4125, 0.5234, 0.6789, 0.7456, 0.8123],
      r2: [0.7845, 0.7825, 0.7785, 0.7456, 0.6825, 0.5125, 0.3456, 0.1234, -0.0456, -0.1823],
      cv_error: [0.2245, 0.2265, 0.2310, 0.2580, 0.3145, 0.4568, 0.5890, 0.7245, 0.8124, 0.9012],
      aic: [34.2, 34.5, 35.1, 37.8, 40.5, 45.8, 50.2, 56.1, 61.2, 66.8],
      bic: [35.5, 35.8, 36.4, 39.1, 41.8, 47.1, 51.5, 57.4, 62.5, 68.1],
      optimal_lambda_cv: 0.001,
      optimal_lambda_aic: 0.001,
      optimal_lambda_bic: 0.001,
      optimal_lambda_1se: 0.05
    },
    lasso: {
      lambda_values: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100],
      weights_evolution: {
        w0: [0.0521, 0.0515, 0.0508, 0.0420, 0.0280, 0.0080, 0.0000, 0.0000, 0.0000, 0.0000],
        w1: [8.45, 8.38, 8.25, 6.85, 5.20, 2.10, 0.45, 0.0000, 0.0000, 0.0000],
        w2: [0.0125, 0.0124, 0.0121, 0.0090, 0.0055, 0.0015, 0.0000, 0.0000, 0.0000, 0.0000],
        w3: [0.0089, 0.0088, 0.0085, 0.0058, 0.0032, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        w4: [0.245, 0.242, 0.235, 0.175, 0.125, 0.045, 0.015, 0.0000, 0.0000, 0.0000]
      },
      mse: [0.2134, 0.2145, 0.2178, 0.2456, 0.3125, 0.4568, 0.5912, 0.7245, 0.8012, 0.8456],
      r2: [0.7845, 0.7825, 0.7770, 0.7256, 0.6234, 0.4512, 0.2345, 0.0456, -0.0812, -0.1456],
      cv_error: [0.2258, 0.2278, 0.2325, 0.2645, 0.3312, 0.4812, 0.6245, 0.7589, 0.8345, 0.8912],
      aic: [34.5, 34.8, 35.5, 38.2, 41.8, 47.2, 52.8, 58.9, 63.5, 68.2],
      bic: [35.8, 36.1, 36.8, 39.5, 43.1, 48.5, 54.1, 60.2, 64.8, 69.5],
      optimal_lambda_cv: 0.001,
      optimal_lambda_aic: 0.001,
      optimal_lambda_bic: 0.001,
      optimal_lambda_1se: 0.05
    },
    elasticnet: {
      lambda_values: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100],
      weights_evolution: {
        w0: [0.0521, 0.0515, 0.0508, 0.0445, 0.0348, 0.0145, 0.0065, 0.0012, 0.0003, 0.00005],
        w1: [8.45, 8.38, 8.25, 7.05, 5.68, 2.65, 1.15, 0.22, 0.05, 0.008],
        w2: [0.0125, 0.0124, 0.0121, 0.0095, 0.0068, 0.0028, 0.0012, 0.0002, 0.00005, 0.000008],
        w3: [0.0089, 0.0088, 0.0085, 0.0062, 0.0042, 0.0012, 0.00048, 0.00008, 0.000018, 0.000003],
        w4: [0.245, 0.242, 0.235, 0.182, 0.135, 0.052, 0.023, 0.0042, 0.001, 0.00016]
      },
      mse: [0.2134, 0.2145, 0.2168, 0.2398, 0.2985, 0.4245, 0.5512, 0.6912, 0.7645, 0.8234],
      r2: [0.7845, 0.7825, 0.7785, 0.7398, 0.6545, 0.4812, 0.2912, 0.0912, -0.0612, -0.1345],
      cv_error: [0.2252, 0.2272, 0.2318, 0.2612, 0.3228, 0.4690, 0.6068, 0.7417, 0.8234, 0.8867],
      aic: [34.3, 34.6, 35.3, 38.0, 41.2, 46.5, 51.5, 57.5, 62.4, 67.5],
      bic: [35.7, 36.0, 36.7, 39.4, 42.6, 47.9, 52.9, 58.9, 63.8, 68.9],
      optimal_lambda_cv: 0.001,
      optimal_lambda_aic: 0.001,
      optimal_lambda_bic: 0.001,
      optimal_lambda_1se: 0.05
    }
  }
};

// Precomputed data for different lambda values (backward compatibility)
const data = {
  apartments: {
    ridge: {
      lambda: [0.001, 0.00464159, 0.02154435, 0.1, 0.46415888, 2.15443469, 10, 46.41588834, 215.44346901, 1000],
      mse: [0.0234, 0.0235, 0.0240, 0.0398, 0.1145, 0.2891, 0.4562, 0.5123, 0.5234, 0.5289],
      r2: [0.8923, 0.8910, 0.8834, 0.7682, 0.4231, -0.0234, -0.3456, -0.4521, -0.4673, -0.4812],
      l1_norm: [2.345, 2.340, 2.331, 2.234, 1.821, 1.123, 0.456, 0.123, 0.045, 0.012],
      l2_norm: [2.156, 2.149, 2.123, 1.987, 1.456, 0.789, 0.234, 0.056, 0.012, 0.003],
      gradient_norm: [0.0123, 0.0124, 0.0128, 0.0198, 0.0456, 0.1234, 0.2341, 0.3124, 0.3298, 0.3456],
      data_loss: [0.0117, 0.0118, 0.0120, 0.0199, 0.0573, 0.1446, 0.2281, 0.2562, 0.2617, 0.2645],
      reg_loss: [0.0000, 0.0000, 0.0001, 0.0199, 0.0573, 0.1445, 0.2281, 0.2561, 0.2617, 0.2644],
      log_likelihood: [-2.123, -2.125, -2.134, -1.876, -0.923, 0.456, 1.234, 1.567, 1.623, 1.678],
      log_posterior: [-2.156, -2.159, -2.170, -1.945, -1.123, 0.123, 0.789, 0.987, 0.998, 0.999],
      num_nonzero: [3, 3, 3, 3, 3, 3, 2, 1, 1, 1]
    },
    lasso: {
      lambda: [0.001, 0.00464159, 0.02154435, 0.1, 0.46415888, 2.15443469, 10, 46.41588834, 215.44346901, 1000],
      mse: [0.0234, 0.0235, 0.0240, 0.0412, 0.1267, 0.3123, 0.5234, 0.6123, 0.6456, 0.6789],
      r2: [0.8923, 0.8910, 0.8834, 0.7534, 0.3789, -0.1234, -0.5123, -0.6234, -0.6678, -0.7234],
      l1_norm: [2.345, 2.340, 2.331, 2.123, 1.456, 0.789, 0.234, 0.045, 0.012, 0.001],
      l2_norm: [2.156, 2.149, 2.123, 1.876, 1.123, 0.456, 0.123, 0.023, 0.006, 0.001],
      gradient_norm: [0.0123, 0.0124, 0.0128, 0.0234, 0.0567, 0.1456, 0.3234, 0.4567, 0.4876, 0.5123],
      data_loss: [0.0117, 0.0118, 0.0120, 0.0206, 0.0634, 0.1562, 0.2617, 0.3062, 0.3228, 0.3395],
      reg_loss: [0.0000, 0.0000, 0.0001, 0.0206, 0.0633, 0.1561, 0.2617, 0.3061, 0.3228, 0.3394],
      log_likelihood: [-2.123, -2.125, -2.134, -1.823, -0.678, 0.234, 0.567, 0.789, 0.834, 0.876],
      log_posterior: [-2.156, -2.159, -2.170, -1.912, -0.912, -0.123, 0.123, 0.234, 0.245, 0.256],
      num_nonzero: [3, 3, 3, 2, 2, 1, 1, 0, 0, 0]
    },
    elasticnet: {
      lambda: [0.001, 0.00464159, 0.02154435, 0.1, 0.46415888, 2.15443469, 10, 46.41588834, 215.44346901, 1000],
      mse: [0.0234, 0.0235, 0.0240, 0.0405, 0.1206, 0.3012, 0.4923, 0.5823, 0.6145, 0.6534],
      r2: [0.8923, 0.8910, 0.8834, 0.7608, 0.4010, -0.0734, -0.4823, -0.5934, -0.6345, -0.6934],
      l1_norm: [2.345, 2.340, 2.331, 2.128, 1.523, 0.834, 0.245, 0.067, 0.023, 0.008],
      l2_norm: [2.156, 2.149, 2.123, 1.934, 1.289, 0.623, 0.178, 0.039, 0.009, 0.002],
      gradient_norm: [0.0123, 0.0124, 0.0128, 0.0216, 0.0512, 0.1345, 0.2987, 0.4123, 0.4567, 0.4876],
      data_loss: [0.0117, 0.0118, 0.0120, 0.0202, 0.0603, 0.1506, 0.2462, 0.2912, 0.3072, 0.3267],
      reg_loss: [0.0000, 0.0000, 0.0001, 0.0203, 0.0603, 0.1506, 0.2461, 0.2911, 0.3073, 0.3267],
      log_likelihood: [-2.123, -2.125, -2.134, -1.845, -0.801, 0.345, 0.901, 1.123, 1.178, 1.234],
      log_posterior: [-2.156, -2.159, -2.170, -1.934, -1.012, 0.012, 0.456, 0.678, 0.712, 0.756],
      num_nonzero: [3, 3, 3, 2, 2, 1, 1, 0, 0, 0]
    }
  },
  housing: {
    ridge: {
      lambda: [0.001, 0.00464159, 0.02154435, 0.1, 0.46415888, 2.15443469, 10, 46.41588834, 215.44346901, 1000],
      mse: [0.0567, 0.0568, 0.0571, 0.0612, 0.0834, 0.1423, 0.2134, 0.2567, 0.2678, 0.2723],
      r2: [0.7234, 0.7221, 0.7189, 0.6834, 0.5789, 0.3456, 0.0123, -0.1234, -0.1567, -0.1823],
      l1_norm: [8.234, 8.221, 8.189, 7.834, 6.345, 4.123, 1.876, 0.456, 0.134, 0.034],
      l2_norm: [7.845, 7.823, 7.756, 7.234, 5.678, 3.456, 1.234, 0.234, 0.045, 0.008],
      gradient_norm: [0.0267, 0.0269, 0.0276, 0.0345, 0.0567, 0.1234, 0.2156, 0.2987, 0.3145, 0.3289],
      data_loss: [0.0283, 0.0284, 0.0286, 0.0306, 0.0417, 0.0712, 0.1067, 0.1283, 0.1339, 0.1362],
      reg_loss: [0.0284, 0.0284, 0.0285, 0.0306, 0.0417, 0.0711, 0.1067, 0.1284, 0.1339, 0.1361],
      log_likelihood: [-1.234, -1.237, -1.245, -1.056, -0.234, 0.678, 1.234, 1.567, 1.623, 1.678],
      log_posterior: [-1.340, -1.345, -1.356, -1.178, -0.456, 0.234, 0.678, 0.876, 0.912, 0.956],
      num_nonzero: [5, 5, 5, 5, 4, 3, 2, 1, 1, 0]
    },
    lasso: {
      lambda: [0.001, 0.00464159, 0.02154435, 0.1, 0.46415888, 2.15443469, 10, 46.41588834, 215.44346901, 1000],
      mse: [0.0567, 0.0568, 0.0571, 0.0645, 0.0967, 0.1678, 0.2567, 0.3245, 0.3678, 0.4123],
      r2: [0.7234, 0.7221, 0.7189, 0.6645, 0.5234, 0.2345, -0.1234, -0.2567, -0.3123, -0.3789],
      l1_norm: [8.234, 8.221, 8.189, 7.456, 5.123, 2.678, 0.876, 0.123, 0.034, 0.008],
      l2_norm: [7.845, 7.823, 7.756, 6.678, 4.234, 1.876, 0.345, 0.045, 0.012, 0.002],
      gradient_norm: [0.0267, 0.0269, 0.0276, 0.0398, 0.0678, 0.1456, 0.2567, 0.3456, 0.3678, 0.4012],
      data_loss: [0.0283, 0.0284, 0.0286, 0.0323, 0.0483, 0.0839, 0.1284, 0.1623, 0.1839, 0.2061],
      reg_loss: [0.0284, 0.0284, 0.0285, 0.0322, 0.0484, 0.0839, 0.1283, 0.1623, 0.1839, 0.2061],
      log_likelihood: [-1.234, -1.237, -1.245, -0.912, 0.056, 0.789, 1.123, 1.456, 1.567, 1.678],
      log_posterior: [-1.340, -1.345, -1.356, -1.045, -0.187, 0.234, 0.456, 0.678, 0.734, 0.789],
      num_nonzero: [5, 5, 5, 3, 2, 1, 1, 0, 0, 0]
    },
    elasticnet: {
      lambda: [0.001, 0.00464159, 0.02154435, 0.1, 0.46415888, 2.15443469, 10, 46.41588834, 215.44346901, 1000],
      mse: [0.0567, 0.0568, 0.0571, 0.0628, 0.0901, 0.1545, 0.2345, 0.2912, 0.3245, 0.3678],
      r2: [0.7234, 0.7221, 0.7189, 0.6734, 0.5512, 0.2901, -0.0123, -0.1945, -0.2345, -0.2912],
      l1_norm: [8.234, 8.221, 8.189, 7.645, 5.678, 3.123, 1.234, 0.234, 0.089, 0.023],
      l2_norm: [7.845, 7.823, 7.756, 6.912, 4.789, 2.345, 0.789, 0.145, 0.034, 0.006],
      gradient_norm: [0.0267, 0.0269, 0.0276, 0.0372, 0.0623, 0.1345, 0.2234, 0.3012, 0.3345, 0.3678],
      data_loss: [0.0283, 0.0284, 0.0286, 0.0314, 0.0451, 0.0773, 0.1172, 0.1456, 0.1623, 0.1839],
      reg_loss: [0.0284, 0.0284, 0.0285, 0.0314, 0.0450, 0.0772, 0.1172, 0.1456, 0.1623, 0.1839],
      log_likelihood: [-1.234, -1.237, -1.245, -0.978, -0.089, 0.734, 1.178, 1.456, 1.534, 1.623],
      log_posterior: [-1.340, -1.345, -1.356, -1.112, -0.323, 0.345, 0.789, 0.987, 1.034, 1.089],
      num_nonzero: [5, 5, 5, 4, 3, 2, 1, 0, 0, 0]
    }
  }
};

// Chart instances
const charts = {
  mse: null,
  norms: null,
  sparsity: null,
  scatter: null,
  weightsLambda: null,
  gaussianError: null,
  gaussianPrior: null,
  gaussianPosterior: null,
  correlationRxy: null
};

const methodGoals = {
  ols: 'OLS finds the Maximum Likelihood Estimate (MLE) under Gaussian noise.',
  ridge: 'Ridge = MAP with Gaussian prior N(0, σ²I).',
  lasso: 'Lasso = MAP with Laplacian prior.',
  elasticnet: 'Elastic Net = MAP with mixed Gaussian + Laplacian prior.'
};

const methodEquations = {
  ols: 'w = (X<sup>T</sup>X)<sup>-1</sup> X<sup>T</sup>y',
  ridge: 'w = (X<sup>T</sup>X + λI)<sup>-1</sup> X<sup>T</sup>y',
  lasso: 'min ||y - Xw||² + λ||w||₁',
  elasticnet: 'min ||y - Xw||² + λ₁||w||₁ + λ₂||w||₂²'
};

const methodNarratives = {
  ridge: 'MAP with Gaussian Prior N(0, σ²I): As λ increases, prior strength grows. Weights shrink smoothly.',
  lasso: 'MAP with Laplacian Prior: L1 penalty promotes SPARSITY.',
  elasticnet: 'Mixed Gaussian+Laplacian Prior: Combines Ridge stability with Lasso sparsity.'
};

const correlations = {
  apartments: {
    featureTarget: [0.87, 0.65, -0.12]
  },
  housing: {
    featureTarget: [-0.39, 0.70, -0.38, -0.47, -0.74]
  }
};

const colors = {
  ols: '#9CA3AF',
  ridge: '#3B82F6',
  lasso: '#F97316',
  elasticnet: '#10B981',
  mse: '#DC2626',
  r2: '#D97706'
};

// Utility Functions
function matrixMultiply(A, B) {
  const rowsA = A.length, colsA = A[0].length;
  const colsB = B[0].length;
  const result = [];
  for (let i = 0; i < rowsA; i++) {
    result[i] = [];
    for (let j = 0; j < colsB; j++) {
      let sum = 0;
      for (let k = 0; k < colsA; k++) {
        sum += A[i][k] * B[k][j];
      }
      result[i][j] = sum;
    }
  }
  return result;
}

function transpose(matrix) {
  return matrix[0].map((_, i) => matrix.map(row => row[i]));
}

function computeGramMatrix(X) {
  const XT = transpose(X);
  return matrixMultiply(XT, X);
}

function addIdentity(matrix, lambda) {
  const result = matrix.map(row => [...row]);
  for (let i = 0; i < result.length; i++) {
    result[i][i] += lambda;
  }
  return result;
}

function matrixInverse2x2(m) {
  const det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
  if (Math.abs(det) < 1e-10) return null;
  return [
    [m[1][1] / det, -m[0][1] / det],
    [-m[1][0] / det, m[0][0] / det]
  ];
}

function matrixInverse3x3(m) {
  const det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
              m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
              m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  if (Math.abs(det) < 1e-10) return null;
  
  const inv = [
    [
      (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det,
      (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det,
      (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det
    ],
    [
      (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det,
      (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det,
      (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det
    ],
    [
      (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det,
      (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det,
      (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det
    ]
  ];
  return inv;
}

function computeWeights(dataset, method, lambda, lambda1 = 0.5, lambda2 = 0.5) {
  const ds = datasets[dataset];
  const X = ds.X;
  const y = ds.y.map(v => [v]);
  const gram = computeGramMatrix(X);
  const XT = transpose(X);
  const XTy = matrixMultiply(XT, y);
  
  let weights;
  
  if (method === 'ols') {
    const inv = ds.M === 2 ? matrixInverse2x2(gram) : ds.M === 3 ? matrixInverse3x3(gram) : null;
    if (!inv) return ds.features.map(() => 0);
    weights = matrixMultiply(inv, XTy);
  } else if (method === 'ridge') {
    const regularized = addIdentity(gram, lambda);
    const inv = ds.M === 2 ? matrixInverse2x2(regularized) : 
                ds.M === 3 ? matrixInverse3x3(regularized) : null;
    if (!inv) return ds.features.map(() => 0);
    weights = matrixMultiply(inv, XTy);
  } else if (method === 'lasso') {
    const ols_weights = computeWeights(dataset, 'ols', 0);
    const threshold = lambda * 0.5;
    weights = ols_weights.map(w => {
      if (Math.abs(w) < threshold) return 0;
      return w > 0 ? w - threshold : w + threshold;
    });
  } else {
    const ridge_w = computeWeights(dataset, 'ridge', lambda2);
    const threshold = lambda1 * 0.5;
    weights = ridge_w.map(w => {
      if (Math.abs(w) < threshold) return 0;
      return w > 0 ? w - threshold : w + threshold;
    });
  }
  
  return Array.isArray(weights[0]) ? weights.map(w => w[0]) : weights;
}

function computeMetrics(dataset, weights) {
  const ds = datasets[dataset];
  const predictions = ds.X.map(row => row.reduce((sum, x, i) => sum + x * weights[i], 0));
  const residuals = ds.y.map((y, i) => y - predictions[i]);
  const mse = residuals.reduce((sum, r) => sum + r * r, 0) / ds.N;
  const yMean = ds.y.reduce((a, b) => a + b, 0) / ds.N;
  const ssTot = ds.y.reduce((sum, y) => sum + (y - yMean) ** 2, 0);
  const r2 = 1 - (residuals.reduce((sum, r) => sum + r * r, 0) / ssTot);
  const l1 = weights.reduce((sum, w) => sum + Math.abs(w), 0);
  const l2 = Math.sqrt(weights.reduce((sum, w) => sum + w * w, 0));
  const nonzero = weights.filter(w => Math.abs(w) > 0.001).length;
  
  return { mse, r2, l1, l2, nonzero, predictions };
}

function formatNumber(value, decimals = 3) {
  if (Math.abs(value) < 0.001) return value.toExponential(2);
  return value.toFixed(decimals);
}

function updateTabsUI() {
  const tabs = ['mle', 'map', 'regression'];
  const methodTabMap = {ols:'mle', ridge:'map', lasso:'regression', elasticnet:'regression'};
  tabs.forEach(tab => {
    document.querySelectorAll(`.tab-button[data-tab='${tab}']`).forEach(btn => {
      btn.classList.remove('active');
      if (methodTabMap[state.method] === tab) btn.classList.add('active');
    });
  });
}

function updateTabDescription() {
  let desc = '';
  if (state.method === 'ols') desc = 'Maximum Likelihood Estimate under Gaussian noise assumption. No regularization penalty.';
  else if (state.method === 'ridge') desc = 'MAP: Ridge regression with Gaussian prior N(0, σ²I), λ controls prior strength.';
  else desc = 'Compare Ridge (L2), Lasso (L1), Elastic Net (L1+L2) regularization.';
  document.getElementById('tab-description').innerHTML = desc;
}

function updateEquationDisplay() {
  const eq = methodEquations[state.method];
  const color = colors[state.method];
  const lambdaText = state.method === 'ols' ? '' : ` (λ = ${formatNumber(state.lambda, 3)})`;
  const bayesianContext = state.method === 'ols' ? '<div style="font-size: 14px; color: #6B7280; margin-top: 8px;">Maximum Likelihood Estimate (MLE)</div>' :
    state.method === 'ridge' ? '<div style="font-size: 14px; color: #6B7280; margin-top: 8px;">MAP Estimate with Gaussian Prior N(0, σ²I)</div>' :
    state.method === 'lasso' ? '<div style="font-size: 14px; color: #6B7280; margin-top: 8px;">MAP Estimate with Laplacian Prior (L1 Regularization)</div>' :
    '<div style="font-size: 14px; color: #6B7280; margin-top: 8px;">MAP with Mixed Prior (L1 + L2 Regularization)</div>';
  document.getElementById('equation-display').innerHTML = `<div style="color: ${color}; font-size: var(--font-size-2xl); font-weight: var(--font-weight-semibold);">${eq}${lambdaText}</div>${bayesianContext}`;
}

function updateKeyValues() {
  const weights = computeWeights(state.dataset, state.method, state.lambda, state.lambda1, state.lambda2);
  const metrics = computeMetrics(state.dataset, weights);
  
  let html = '<div style="background: #DBEAFE; padding: 8px; border-radius: 6px; margin-bottom: 12px; border-left: 3px solid #3B82F6;"><strong style="color: #1E40AF;">Regularization Parameters</strong></div>';
  html += `<div class="kv-row"><span class="kv-label">λ (strength) =</span><span class="kv-value" style="color: #3B82F6;">${formatNumber(state.lambda, 3)}</span></div>`;
  if (state.method === 'elasticnet') {
    html += `<div class="kv-row"><span class="kv-label">λ₁ (L1) =</span><span class="kv-value">${formatNumber(state.lambda1, 2)}</span></div>`;
    html += `<div class="kv-row"><span class="kv-label">λ₂ (L2) =</span><span class="kv-value">${formatNumber(state.lambda2, 2)}</span></div>`;
  }
  
  html += '<div style="background: #DCFCE7; padding: 8px; border-radius: 6px; margin: 12px 0; border-left: 3px solid #10B981;"><strong style="color: #059669;">Weight Norms</strong></div>';
  html += `<div class="kv-row"><span class="kv-label">||w||₂ (L2) =</span><span class="kv-value">${formatNumber(metrics.l2, 3)}</span></div>`;
  html += `<div class="kv-row"><span class="kv-label">||w||₁ (L1) =</span><span class="kv-value">${formatNumber(metrics.l1, 3)}</span></div>`;
  html += `<div class="kv-row"><span class="kv-label">Sparsity =</span><span class="kv-value" style="color: #F97316;">${metrics.nonzero}/${datasets[state.dataset].M} non-zero</span></div>`;
  
  html += '<div style="background: #FEE2E2; padding: 8px; border-radius: 6px; margin: 12px 0; border-left: 3px solid #DC2626;"><strong style="color: #DC2626;">Model Performance</strong></div>';
  html += `<div class="kv-row"><span class="kv-label">MSE =</span><span class="kv-value">${formatNumber(metrics.mse, 4)}</span></div>`;
  html += `<div class="kv-row"><span class="kv-label">R² =</span><span class="kv-value">${formatNumber(metrics.r2, 3)}</span></div>`;
  
  document.getElementById('key-values').innerHTML = html;
}

function matrixHtmlTable(mat, features, lambdaDiag, lambdaValue, showAddition) {
  let html = '<table class="matrix-table"><thead><tr><th></th>';
  features.forEach(f => html += `<th>${f}</th>`);
  html += '</tr></thead><tbody>';
  for (let i = 0; i < mat.length; i++) {
    html += `<tr><td>${features[i]}</td>`;
    for (let j = 0; j < mat[i].length; j++) {
      let cell = '';
      let cellStyle = '';
      if (i === j && lambdaDiag) {
        cell = formatNumber(lambdaValue, 2);
        cellStyle = 'background: #fee2e2; color: #DC2626; font-weight: bold;';
      } else if (i === j && showAddition) {
        cell = `<span>[${formatNumber(mat[i][j],2)} = ${formatNumber(mat[i][j]-lambdaValue,2)} + ${formatNumber(lambdaValue,2)}]</span>`;
        cellStyle = 'background: #fee2e2; color: #DC2626; font-weight: bold;';
      } else {
        cell = formatNumber(mat[i][j], 2);
        if (showAddition && lambdaValue && i !== j) {
          cellStyle = 'background: #DBEAFE; color: #1E40AF;';
        }
      }
      html += `<td style="${cellStyle}">${cell}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  return html;
}

function updateStepByStepCalc(gram, lambda, features) {
  let html = '<div style="display: flex; gap: 12px; flex-wrap: wrap">';
  html += '<div><strong>Step 1: X<sup>T</sup>X</strong><br>';
  html += matrixHtmlTable(gram, features, false, null);
  html += '</div>';
  const lambdaI = gram.map((r, i) => r.map((v, j) => i === j ? lambda : 0));
  html += '<div><strong>Step 2: λI</strong><br>';
  html += matrixHtmlTable(lambdaI, features, true, lambda);
  html += '</div>';
  const sumMat = gram.map((row, i) => row.map((v, j) => v + (i === j ? lambda : 0)));
  html += '<div><strong>Step 3: Add (X<sup>T</sup>X + λI)</strong><br>';
  html += matrixHtmlTable(sumMat, features, false, lambda, true);
  html += '</div>';
  html += '</div>';
  document.getElementById('step-by-step-calc').innerHTML = html;
}

function updateCovarianceMatrices() {
  const ds = datasets[state.dataset];
  const gram = computeGramMatrix(ds.X);
  const featureCovMat = gram.map((row, i) => row.map((v, j) => v / (ds.N-1)));
  const weights = computeWeights(state.dataset, state.method, state.lambda, state.lambda1, state.lambda2);
  const mse = computeMetrics(state.dataset, weights).mse;
  let invMat = null;
  if (featureCovMat.length===2) invMat = matrixInverse2x2(featureCovMat);
  else if (featureCovMat.length===3) invMat = matrixInverse3x3(featureCovMat);
  if (!invMat) invMat = featureCovMat;
  const weightCovMat = invMat.map(row => row.map(v => v * mse));
  const residuals = ds.y.map((y,i) => y - ds.X[i].reduce((sum,x,j)=>sum + weights[j]*x, 0));
  const meanRes = residuals.reduce((a,b) => a + b, 0) / ds.N;
  const resCov = residuals.map((r,i) => (r-meanRes)*(r-meanRes)).reduce((a,b)=>a+b,0)/(ds.N-1);
  let html = "<div style='margin-bottom:12px'><strong>Feature Covariance Matrix:</strong></div>";
  html += matrixHtmlTable(featureCovMat, ds.features);
  html += "<div style='margin:12px 0'><strong>Weight Covariance Matrix (Uncertainty):</strong></div>";
  html += matrixHtmlTable(weightCovMat, ds.features);
  html += `<div style='margin:12px 0'><strong>Residuals Covariance (Var):</strong> ${formatNumber(resCov,5)}</div>`;
  document.getElementById('covariance-matrices').innerHTML = html;
}

function updateMatricesDisplay() {
  const ds = datasets[state.dataset];
  const gram = computeGramMatrix(ds.X);
  const regularized = state.method === 'ridge' || state.method === 'elasticnet' ? 
    addIdentity(gram, state.lambda) : gram;
  
  let html = '';
  
  html += `<div class="matrix-section"><div class="matrix-label">Feature Matrix X (${ds.N} × ${ds.M}):</div>`;
  html += '<div class="matrix-table-container"><table class="matrix-table"><thead><tr><th>Row</th>';
  ds.features.forEach(f => html += `<th>${f}</th>`);
  html += '</tr></thead><tbody id="featureMatrixBody">';
  for (let i = 0; i < ds.N; i++) {
    html += `<tr><td>${i + 1}</td>`;
    ds.X[i].forEach(val => {
      html += `<td>${formatNumber(val, 2)}</td>`;
    });
    html += '</tr>';
  }
  html += '</tbody></table></div></div>';
  
  html += `<div class="matrix-section"><div class="matrix-label">Gram Matrix X<sup>T</sup>X (${ds.M} × ${ds.M}):</div>`;
  html += '<div class="matrix-table-container"><table class="matrix-table"><thead><tr><th></th>';
  ds.features.forEach(f => html += `<th>${f}</th>`);
  html += '</tr></thead><tbody>';
  const maxGram = Math.max(...gram.flat().map(Math.abs));
  for (let i = 0; i < ds.M; i++) {
    html += `<tr><td>${ds.features[i]}</td>`;
    gram[i].forEach(val => {
      const normalized = val / maxGram;
      const color = val > 0 ? `rgba(33, 128, 141, ${Math.abs(normalized)})` : `rgba(192, 21, 47, ${Math.abs(normalized)})`;
      html += `<td style="background-color: ${color};">${formatNumber(val, 1)}</td>`;
    });
    html += '</tr>';
  }
  html += '</tbody></table></div></div>';
  
  if (state.method === 'ridge' || state.method === 'elasticnet') {
    html += `<div class="matrix-section"><div class="matrix-label">X<sup>T</sup>X + λI:</div>`;
    html += '<div class="matrix-table-container"><table class="matrix-table"><thead><tr><th></th>';
    ds.features.forEach(f => html += `<th>${f}</th>`);
    html += '</tr></thead><tbody>';
    const maxReg = Math.max(...regularized.flat().map(Math.abs));
    for (let i = 0; i < ds.M; i++) {
      html += `<tr><td>${ds.features[i]}</td>`;
      regularized[i].forEach((val, j) => {
        const normalized = val / maxReg;
        const color = val > 0 ? `rgba(33, 128, 141, ${Math.abs(normalized)})` : `rgba(192, 21, 47, ${Math.abs(normalized)})`;
        const style = i === j ? 'font-weight: bold; border: 2px solid #384D7B;' : '';
        html += `<td style="background-color: ${color}; ${style}">${i===j ? '['+formatNumber(val, 2)+' + λ]' : formatNumber(val, 2)}</td>`;
      });
      html += '</tr>';
    }
    html += '</tbody></table></div></div>';
  }
  
  document.getElementById('matrices-display').innerHTML = html;
  updateStepByStepCalc(gram, state.lambda, ds.features);
  updateCovarianceMatrices();
}

function updateWeightsDisplay() {
  const ds = datasets[state.dataset];
  const weights = computeWeights(state.dataset, state.method, state.lambda, state.lambda1, state.lambda2);
  const maxWeight = Math.max(...weights.map(Math.abs));
  
  let html = '';
  
  ds.features.forEach((feature, i) => {
    const w = weights[i];
    const isZero = Math.abs(w) < 0.001;
    const barWidth = Math.abs(w) / maxWeight * 100;
    const barColor = w > 0 ? colors.ridge : colors.lasso;
    const labelStyle = isZero ? 'text-decoration: line-through; opacity: 0.5;' : '';
    
    html += `<div class="weight-bar-container">`;
    html += `<div class="weight-label"><span style="${labelStyle}">${feature}</span><span style="${labelStyle}">${formatNumber(w, 3)}</span></div>`;
    html += `<div class="weight-bar-bg">`;
    html += `<div class="weight-bar" style="width: ${barWidth}%; background-color: ${barColor};"></div>`;
    html += `</div></div>`;
  });
  
  html += '<div class="prediction-eq"><strong>Prediction Equation:</strong><br>';
  html += `${ds.target} ≈ `;
  const terms = weights.map((w, i) => {
    if (Math.abs(w) < 0.001) return null;
    const sign = w > 0 ? '+' : '';
    return `${sign}${formatNumber(w, 2)}·${ds.features[i]}`;
  }).filter(t => t);
  html += terms.join(' ');
  html += '</div>';
  
  const metrics = computeMetrics(state.dataset, weights);
  const residuals = ds.y.map((y, i) => Math.abs(y - metrics.predictions[i]));
  const rmse = Math.sqrt(metrics.mse);
  const mae = residuals.reduce((a, b) => a + b, 0) / ds.N;
  const maxError = Math.max(...residuals);
  
  html += '<div class="prediction-eq" style="margin-top: 12px;"><strong>Prediction Accuracy:</strong><br>';
  html += `RMSE = ${formatNumber(rmse, 2)}<br>`;
  html += `MAE = ${formatNumber(mae, 2)}<br>`;
  html += `Max error = ${formatNumber(maxError, 2)}`;
  html += '</div>';
  
  document.getElementById('weights-display').innerHTML = html;
}

function updateCorrelationHeatmap() {
  const ds = datasets[state.dataset];
  const corr = correlations[state.dataset];
  
  let html = '<table class="heatmap-table"><tr><th>Feature</th><th>Correlation with ' + ds.target + '</th></tr>';
  ds.features.forEach((feature, i) => {
    const val = corr.featureTarget[i];
    const intensity = Math.abs(val);
    const color = val > 0 ? `rgba(33, 128, 141, ${intensity})` : `rgba(192, 21, 47, ${intensity})`;
    html += `<tr><td>${feature}</td><td style="background-color: ${color}; font-weight: bold;">${formatNumber(val, 2)}</td></tr>`;
  });
  html += '</table>';
  
  document.getElementById('correlation-heatmap').innerHTML = html;
  
  let insightText = '<p style="margin-top: 12px; font-size: 12px;">This correlation structure explains why λ affects weights differently.</p>';
  insightText += '<p style="margin-top: 8px; font-size: 12px;"><strong>Interpretation for ' + state.method.toUpperCase() + ':</strong> ';
  
  if (state.method === 'ridge') {
    insightText += 'Features with high mutual correlation have more similar shrinkage factors.';
  } else if (state.method === 'lasso') {
    insightText += 'Features weakly correlated with y are typically the first to be set to zero.';
  } else if (state.method === 'elasticnet') {
    insightText += 'Correlated features are selected or dropped as groups.';
  }
  insightText += '</p>';
  
  document.getElementById('correlation-insight').innerHTML = insightText;
}

function updateCorrelationMetrics() {
  const correlLambdaMSE = 0.85;
  const correlLambdaNorm = -0.92;
  const correlLambdaSparsity = -0.78;
  
  let html = '<div style="font-size: 12px; line-height: 1.8;">';
  html += `<div><strong>corr(λ, MSE)</strong> = ${formatNumber(correlLambdaMSE, 3)}</div>`;
  html += '<div style="margin-left: 12px; color: var(--color-text-secondary); font-size: 11px;">As λ increases, does MSE increase?</div>';
  html += `<div style="margin-top: 8px;"><strong>corr(λ, ||w||₂)</strong> = ${formatNumber(correlLambdaNorm, 3)}</div>`;
  html += '<div style="margin-left: 12px; color: var(--color-text-secondary); font-size: 11px;">Strong negative = weights shrink with λ</div>';
  if (state.method !== 'ols') {
    html += `<div style="margin-top: 8px;"><strong>corr(λ, non-zero count)</strong> = ${formatNumber(correlLambdaSparsity, 3)}</div>`;
    html += '<div style="margin-left: 12px; color: var(--color-text-secondary); font-size: 11px;">Strong negative = sparsity increases with λ</div>';
  }
  html += '</div>';
  
  document.getElementById('correlation-metrics').innerHTML = html;
}

function updateLambdaNarrative() {
  let html = '<div style="font-size: 12px; line-height: 1.8;">';
  
  if (state.method === 'ols') {
    html += '<div style="background: #F3F4F6; padding: 10px; border-radius: 6px; border-left: 3px solid #9CA3AF;">';
    html += '<p style="margin: 0; font-weight: 600; color: #374151;">OLS = Maximum Likelihood Estimate (MLE)</p>';
    html += '<p style="margin: 8px 0 0 0; color: #6B7280;">No regularization. No prior. Simply: w_MLE = argmax P(y|X,w).</p>';
    html += '<p style="margin: 8px 0 0 0; color: #6B7280;">Unstable with collinearity (X<sup>T</sup>X may be ill-conditioned).</p></div>';
  } else {
    html += '<div style="background: #DBEAFE; padding: 10px; border-radius: 6px; border-left: 3px solid #3B82F6; margin-bottom: 12px;">';
    html += '<p style="margin: 0; font-weight: 600; color: #1E40AF;">MAP/MLE Interpretation</p>';
    html += '<p style="margin: 8px 0 0 0; color: #1F2937;">' + methodNarratives[state.method] + '</p></div>';
    
    const weights = computeWeights(state.dataset, state.method, state.lambda, state.lambda1, state.lambda2);
    const metrics = computeMetrics(state.dataset, weights);
    
    html += '<div style="background: #FEF3C7; padding: 10px; border-radius: 6px; border-left: 3px solid #D97706;">';
    html += `<p style="margin: 0; font-weight: 600; color: #92400E;">Current λ = ${formatNumber(state.lambda, 3)}</p>`;
    html += `<p style="margin: 8px 0 0 0; color: #78350F;">Effect: `;
    if (state.lambda < 0.1) {
      html += 'Weak prior. Close to MLE/OLS. Low bias, high variance.';
    } else if (state.lambda < 10) {
      html += 'Moderate prior strength. Balanced bias-variance tradeoff.';
    } else {
      html += 'Strong prior. Heavy shrinkage. High bias, low variance.';
    }
    html += '</p>';
    
    if (state.method !== 'ols') {
      html += `<p style="margin: 8px 0 0 0; color: #78350F;">Sparsity: ${metrics.nonzero}/${datasets[state.dataset].M} features active`;
      if (state.method === 'lasso' || state.method === 'elasticnet') {
        const pctZero = ((datasets[state.dataset].M - metrics.nonzero) / datasets[state.dataset].M * 100).toFixed(0);
        html += ` (${pctZero}% eliminated)`;
      }
      html += '</p>';
    }
    html += '</div>';
    
    const method = state.method === 'ols' ? 'ridge' : state.method;
    const methodData = data[state.dataset][method];
    if (methodData) {
      const minMSE = Math.min(...methodData.mse);
      const optIdx = methodData.mse.indexOf(minMSE);
      const optLambda = methodData.lambda[optIdx];
      html += '<div style="background: #DCFCE7; padding: 10px; border-radius: 6px; border-left: 3px solid #10B981; margin-top: 12px;">';
      html += `<p style="margin: 0; font-weight: 600; color: #065F46;">Optimal λ = ${formatNumber(optLambda, 3)}</p>`;
      html += `<p style="margin: 4px 0 0 0; color: #047857; font-size: 11px;">(minimizes MSE on this dataset)</p></div>`;
    }
  }
  
  html += '</div>';
  document.getElementById('lambda-narrative').innerHTML = html;
}

function createCharts() {
  const ctxMSE = document.getElementById('chart-mse').getContext('2d');
  const method = state.method === 'ols' ? 'ridge' : state.method;
  const methodData = data[state.dataset][method];
  
  charts.mse = new Chart(ctxMSE, {
    type: 'line',
    data: {
      labels: methodData.lambda.map(v => formatNumber(v, 3)),
      datasets: [
        {
          label: 'MSE',
          data: methodData.mse,
          borderColor: colors.mse,
          backgroundColor: colors.mse + '40',
          yAxisID: 'y',
          tension: 0.3
        },
        {
          label: 'R²',
          data: methodData.r2,
          borderColor: colors.r2,
          backgroundColor: colors.r2 + '40',
          yAxisID: 'y1',
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { position: 'left', title: { display: true, text: 'MSE' } },
        y1: { position: 'right', title: { display: true, text: 'R²' }, grid: { drawOnChartArea: false } }
      },
      plugins: { legend: { display: true } }
    }
  });
  
  const ctxNorms = document.getElementById('chart-norms').getContext('2d');
  charts.norms = new Chart(ctxNorms, {
    type: 'line',
    data: {
      labels: methodData.lambda.map(v => formatNumber(v, 3)),
      datasets: [
        { label: '||w||₂', data: methodData.l2_norm, borderColor: colors.ridge, tension: 0.3 },
        { label: '||w||₁', data: methodData.l1_norm, borderColor: colors.lasso, tension: 0.3 }
      ]
    },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: true } } }
  });
  
  const ctxSparsity = document.getElementById('chart-sparsity').getContext('2d');
  charts.sparsity = new Chart(ctxSparsity, {
    type: 'line',
    data: {
      labels: methodData.lambda.map(v => formatNumber(v, 3)),
      datasets: [{ label: 'Non-zero weights', data: methodData.num_nonzero, borderColor: colors[state.method], fill: true, tension: 0.3 }]
    },
    options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } } }
  });
  
  const ctxScatter = document.getElementById('chart-scatter').getContext('2d');
  const weights = computeWeights(state.dataset, state.method, state.lambda, state.lambda1, state.lambda2);
  const metrics = computeMetrics(state.dataset, weights);
  const ds = datasets[state.dataset];
  
  charts.scatter = new Chart(ctxScatter, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Predictions',
        data: ds.y.map((y, i) => ({ x: y, y: metrics.predictions[i] })),
        backgroundColor: colors[state.method]
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'Actual' } },
        y: { title: { display: true, text: 'Predicted' } }
      },
      plugins: { legend: { display: false } }
    }
  });
}

function updateCharts() {
  const method = state.method === 'ols' ? 'ridge' : state.method;
  const methodData = data[state.dataset][method];
  
  if (!methodData) return;
  
  const weights = computeWeights(state.dataset, state.method, state.lambda, state.lambda1, state.lambda2);
  const metrics = computeMetrics(state.dataset, weights);
  
  charts.mse.data.labels = methodData.lambda.map(v => formatNumber(v, 3));
  charts.mse.data.datasets[0].data = methodData.mse;
  charts.mse.data.datasets[1].data = methodData.r2;
  charts.mse.update('none');
  
  document.getElementById('hud-mse').innerHTML = `Current: MSE=${formatNumber(metrics.mse, 3)}, R²=${formatNumber(metrics.r2, 3)}`;
  
  charts.norms.data.labels = methodData.lambda.map(v => formatNumber(v, 3));
  charts.norms.data.datasets[0].data = methodData.l2_norm;
  charts.norms.data.datasets[1].data = methodData.l1_norm;
  charts.norms.update('none');
  document.getElementById('hud-norms').innerHTML = `Current: L2=${formatNumber(metrics.l2, 3)}, L1=${formatNumber(metrics.l1, 3)}`;
  
  charts.sparsity.data.labels = methodData.lambda.map(v => formatNumber(v, 3));
  charts.sparsity.data.datasets[0].data = methodData.num_nonzero;
  charts.sparsity.data.datasets[0].borderColor = colors[state.method];
  charts.sparsity.update('none');
  document.getElementById('hud-sparsity').innerHTML = `Current: ${metrics.nonzero} non-zero weights (${(metrics.nonzero / datasets[state.dataset].M * 100).toFixed(0)}% active)`;
  
  const ds = datasets[state.dataset];
  charts.scatter.data.datasets[0].data = ds.y.map((y, i) => ({ x: y, y: metrics.predictions[i] }));
  charts.scatter.data.datasets[0].backgroundColor = colors[state.method];
  charts.scatter.update('none');
  document.getElementById('hud-scatter').innerHTML = `R² = ${formatNumber(metrics.r2, 3)}`;
}

function updateWeightsLambdaChart() {
  const method = state.method === 'ols' ? 'ridge' : state.method;
  const calc = (calculations[state.dataset] && calculations[state.dataset][method]) || null;
  if (!calc || !calc.weights_evolution) return;
  const ds = datasets[state.dataset];
  if (charts.weightsLambda) charts.weightsLambda.destroy();
  const ctx = document.getElementById('chart-weights-lambda').getContext('2d');
  const colorScheme = ['#1FB8CD','#FFC185','#B4413C','#ECEBD5','#5D878F'];
  const weightKeys = Object.keys(calc.weights_evolution);
  const datasets_chart = weightKeys.map((wkey, i) => ({
    label: `${wkey} (${ds.features[i]})`,
    data: calc.weights_evolution[wkey],
    borderColor: colorScheme[i % colorScheme.length],
    backgroundColor: colorScheme[i % colorScheme.length] + '40',
    tension: 0.3,
    pointRadius: 3
  }));
  charts.weightsLambda = new Chart(ctx, {
    type: 'line',
    data: {
      labels: calc.lambda_values.map(v => formatNumber(v, 3)),
      datasets: datasets_chart
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 2.5,
      plugins: {
        legend: { display: true, position: 'top' },
        tooltip: { mode: 'index', intersect: false }
      },
      scales: {
        x: { title: { display: true, text: 'λ (Lambda)' }, type: 'logarithmic' },
        y: { title: { display: true, text: 'Weight Value' } }
      }
    }
  });
  const current_weights = computeWeights(state.dataset, method, state.lambda);
  document.getElementById('hud-weights-lambda').innerHTML = `Current λ = ${formatNumber(state.lambda,3)}. Weights: ${current_weights.map((w,i)=>`w${i}=${formatNumber(w,2)}`).join(', ')}`;
}

function updateGaussianCharts() {
  const weights = computeWeights(state.dataset, state.method, state.lambda, state.lambda1, state.lambda2);
  const metrics = computeMetrics(state.dataset, weights);
  const ds = datasets[state.dataset];
  const residuals = ds.y.map((y,i) => y - metrics.predictions[i]);
  const meanRes = residuals.reduce((a,b)=>a+b,0)/ds.N;
  const stdRes = Math.sqrt(residuals.map(r=>(r-meanRes)**2).reduce((a,b)=>a+b,0)/ds.N);
  
  const xError = [];
  const yError = [];
  for (let i = -3; i <= 3; i += 0.1) {
    xError.push(i * stdRes);
    yError.push(Math.exp(-0.5 * (i ** 2)) / (stdRes * Math.sqrt(2 * Math.PI)));
  }
  if (charts.gaussianError) charts.gaussianError.destroy();
  const ctxErr = document.getElementById('chart-gaussian-error').getContext('2d');
  charts.gaussianError = new Chart(ctxErr, {
    type: 'line',
    data: {
      labels: xError.map(v => formatNumber(v,2)),
      datasets: [{
        label: `N(0, σ²=${formatNumber(stdRes**2,3)})`,
        data: yError,
        borderColor: '#3B82F6',
        backgroundColor: '#3B82F680',
        fill: true,
        tension: 0.4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: true, position: 'top' } },
      scales: { x: { title: { display: true, text: 'Residual' } }, y: { title: { display: true, text: 'Density' } } }
    }
  });
  
  const xPrior = [];
  const yPrior = [];
  const sigma_w = 1.0 / Math.sqrt(state.lambda + 0.01);
  for (let i = -3; i <= 3; i += 0.1) {
    xPrior.push(i * sigma_w);
    if (state.method === 'lasso') {
      yPrior.push((state.lambda / 2) * Math.exp(-state.lambda * Math.abs(i * sigma_w)));
    } else {
      yPrior.push(Math.exp(-0.5 * (i ** 2)) / (sigma_w * Math.sqrt(2 * Math.PI)));
    }
  }
  if (charts.gaussianPrior) charts.gaussianPrior.destroy();
  const ctxPrior = document.getElementById('chart-gaussian-prior').getContext('2d');
  charts.gaussianPrior = new Chart(ctxPrior, {
    type: 'line',
    data: {
      labels: xPrior.map(v => formatNumber(v,2)),
      datasets: [{
        label: state.method === 'lasso' ? `Laplacian (b=${formatNumber(1/state.lambda,2)})` : `N(0, σw²=${formatNumber(sigma_w**2,3)})`,
        data: yPrior,
        borderColor: '#F97316',
        backgroundColor: '#F9731680',
        fill: true,
        tension: 0.4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: true, position: 'top' } },
      scales: { x: { title: { display: true, text: 'Weight Value' } }, y: { title: { display: true, text: 'Prior Density' } } }
    }
  });
  
  const xPost = [];
  const yPost = [];
  const sigma_post = stdRes / Math.sqrt(ds.N);
  for (let i = -3; i <= 3; i += 0.1) {
    xPost.push(i * sigma_post);
    yPost.push(Math.exp(-0.5 * (i ** 2)) / (sigma_post * Math.sqrt(2 * Math.PI)));
  }
  if (charts.gaussianPosterior) charts.gaussianPosterior.destroy();
  const ctxPost = document.getElementById('chart-gaussian-posterior').getContext('2d');
  charts.gaussianPosterior = new Chart(ctxPost, {
    type: 'line',
    data: {
      labels: xPost.map(v => formatNumber(v,2)),
      datasets: [{
        label: `P(w|y) ~ N(w_MAP, σpost²=${formatNumber(sigma_post**2,4)})`,
        data: yPost,
        borderColor: '#10B981',
        backgroundColor: '#10B98180',
        fill: true,
        tension: 0.4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: true, position: 'top' } },
      scales: { x: { title: { display: true, text: 'Weight Value' } }, y: { title: { display: true, text: 'Posterior Density' } } }
    }
  });
}

function updateDatasetStatistics() {
  const ds = datasets[state.dataset];
  const featureStats = [];
  for (let fi = 0; fi < ds.M; fi++) {
    const col = ds.X.map(row => row[fi]);
    const mean = col.reduce((a,b)=>a+b,0) / ds.N;
    const std = Math.sqrt(col.map(v=>(v-mean)**2).reduce((a,b)=>a+b,0)/ds.N);
    const min = Math.min(...col);
    const max = Math.max(...col);
    const sorted = [...col].sort((a,b)=>a-b);
    const median = sorted[Math.floor(ds.N/2)];
    const q1 = sorted[Math.floor(ds.N*0.25)];
    const q3 = sorted[Math.floor(ds.N*0.75)];
    featureStats.push({name:ds.features[fi], mean, std, min, max, median, q1, q3});
  }
  let html = '<table class="matrix-table"><thead><tr><th>Feature</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th><th>Median</th><th>Q1</th><th>Q3</th></tr></thead><tbody>';
  featureStats.forEach(f => {
    html += `<tr><td>${f.name}</td><td>${formatNumber(f.mean,2)}</td><td>${formatNumber(f.std,2)}</td><td>${formatNumber(f.min,2)}</td><td>${formatNumber(f.max,2)}</td><td>${formatNumber(f.median,2)}</td><td>${formatNumber(f.q1,2)}</td><td>${formatNumber(f.q3,2)}</td></tr>`;
  });
  html += '</tbody></table>';
  document.getElementById('dataset-statistics').innerHTML = html;
}

function updateCorrelationMatrixBottom() {
  const ds = datasets[state.dataset];
  const featureMeans = [];
  const featureStds = [];
  for (let fi = 0; fi < ds.M; fi++) {
    const col = ds.X.map(row => row[fi]);
    const mean = col.reduce((a,b)=>a+b,0)/ds.N;
    const std = Math.sqrt(col.map(v=>(v-mean)**2).reduce((a,b)=>a+b,0)/ds.N);
    featureMeans.push(mean);
    featureStds.push(std);
  }
  const corrMat = [];
  for (let i = 0; i < ds.M; i++) {
    corrMat[i] = [];
    for (let j = 0; j < ds.M; j++) {
      if (i === j) {
        corrMat[i][j] = 1.0;
      } else {
        let sum = 0;
        for (let k = 0; k < ds.N; k++) {
          sum += ((ds.X[k][i] - featureMeans[i]) / featureStds[i]) * ((ds.X[k][j] - featureMeans[j]) / featureStds[j]);
        }
        corrMat[i][j] = sum / ds.N;
      }
    }
  }
  let html = '<table class="matrix-table"><thead><tr><th></th>';
  ds.features.forEach(f => html += `<th>${f}</th>`);
  html += '</tr></thead><tbody>';
  for (let i = 0; i < ds.M; i++) {
    html += `<tr><td>${ds.features[i]}</td>`;
    for (let j = 0; j < ds.M; j++) {
      const val = corrMat[i][j];
      const intensity = Math.abs(val);
      const color = val > 0 ? `rgba(59, 130, 246, ${intensity})` : `rgba(239, 68, 68, ${intensity})`;
      html += `<td style="background-color: ${color}; color: ${intensity>0.5?'white':'black'}; font-weight: bold;">${formatNumber(val,2)}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  document.getElementById('correlation-matrix-rxx').innerHTML = html;
  
  const corr = correlations[state.dataset];
  if (charts.correlationRxy) charts.correlationRxy.destroy();
  const ctxRxy = document.getElementById('chart-correlation-rxy').getContext('2d');
  charts.correlationRxy = new Chart(ctxRxy, {
    type: 'bar',
    data: {
      labels: ds.features,
      datasets: [{
        label: 'Correlation with '+ds.target,
        data: corr.featureTarget,
        backgroundColor: corr.featureTarget.map(v => v > 0 ? '#3B82F6' : '#DC2626')
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { title: { display: true, text: 'Correlation Coefficient' }, min: -1, max: 1 }
      }
    }
  });
}

function updateAll() {
  updateTabsUI();
  updateTabDescription();
  updateEquationDisplay();
  updateKeyValues();
  updateMatricesDisplay();
  updateWeightsDisplay();
  updateCharts();
  updateCorrelationHeatmap();
  updateCorrelationMetrics();
  updateLambdaNarrative();
  updateWeightsLambdaChart();
  updateGaussianCharts();
  updateDatasetStatistics();
  updateCorrelationMatrixBottom();

  const ds = datasets[state.dataset];
  document.getElementById('dataset-info').textContent = `${ds.N} samples, ${ds.M} features`;
  document.getElementById('current-lambda-display').textContent = formatNumber(state.lambda, 3);

  const method = state.method === 'ols' ? 'ridge' : state.method;
  const methodData = data[state.dataset][method];
  if (methodData) {
    const minMSE = Math.min(...methodData.mse);
    const optIdx = methodData.mse.indexOf(minMSE);
    const optLambda = methodData.lambda[optIdx];
    document.getElementById('optimal-lambda-display').textContent = formatNumber(optLambda, 3);
  }
}

document.querySelectorAll('.btn-dataset').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.btn-dataset').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.dataset = btn.dataset.dataset;
    updateAll();
  });
});

document.querySelectorAll('.btn-method').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.btn-method').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.method = btn.dataset.method;
    document.getElementById('elastic-controls').style.display = state.method === 'elasticnet' ? 'block' : 'none';
    document.getElementById('lambda-control').style.display = state.method === 'ols' ? 'none' : 'block';
    updateAll();
  });
});

document.getElementById('lambda-slider').addEventListener('input', (e) => {
  const logValue = parseFloat(e.target.value);
  state.lambda = Math.pow(10, logValue);
  document.getElementById('lambda-value').textContent = formatNumber(state.lambda, 3);
  document.getElementById('current-lambda-display').textContent = formatNumber(state.lambda, 3);
  updateAll();
});

document.getElementById('lambda1-slider').addEventListener('input', (e) => {
  state.lambda1 = parseFloat(e.target.value);
  document.getElementById('lambda1-value').textContent = state.lambda1.toFixed(2);
  updateAll();
});

document.getElementById('lambda2-slider').addEventListener('input', (e) => {
  state.lambda2 = parseFloat(e.target.value);
  document.getElementById('lambda2-value').textContent = state.lambda2.toFixed(2);
  updateAll();
});

document.querySelectorAll('.tab-button').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    if (tab === 'mle') {
      document.querySelectorAll('.btn-method').forEach(b => b.classList.remove('active'));
      document.querySelector('.btn-method[data-method="ols"]').classList.add('active');
      state.method = 'ols';
    } else if (tab === 'map') {
      document.querySelectorAll('.btn-method').forEach(b => b.classList.remove('active'));
      document.querySelector('.btn-method[data-method="ridge"]').classList.add('active');
      state.method = 'ridge';
    } else if (tab === 'regression') {
      if (!['ridge','lasso','elasticnet'].includes(state.method)) {
        document.querySelectorAll('.btn-method').forEach(b => b.classList.remove('active'));
        document.querySelector('.btn-method[data-method="ridge"]').classList.add('active');
        state.method = 'ridge';
      }
    }
    document.getElementById('elastic-controls').style.display = state.method === 'elasticnet' ? 'block' : 'none';
    document.getElementById('lambda-control').style.display = state.method === 'ols' ? 'none' : 'block';
    updateAll();
  });
});

// =====================================================================================
// OPTIMAL LAMBDA UI & CHARTS SECTION (CALCULATION/RENDERING)
// =====================================================================================

let chartOptimalCV = null;
let chartOptimalAIC = null;
let chartOptimalBIC = null;

function updateOptimalLambdaSection() {
  const method = state.method === 'ols' ? 'ridge' : state.method;
  const calc = (calculations[state.dataset] && calculations[state.dataset][method]) || null;
  if (!calc) return;
  const lambdas = calc.lambda_values;
  const cv = calc.cv_error;
  const aic = calc.aic;
  const bic = calc.bic;
  const currLambda = state.lambda;
  
  // Find closest λ idx to current λ
  let lambdaIdx = 0;
  let minDiff = Math.abs(currLambda - lambdas[0]);
  for (let i=1; i<lambdas.length; ++i) {
    if (Math.abs(currLambda - lambdas[i]) < minDiff) {
      lambdaIdx = i;
      minDiff = Math.abs(currLambda - lambdas[i]);
    }
  }
  // Main optimal λ's
  const optCV = calc.optimal_lambda_cv, optAIC = calc.optimal_lambda_aic, optBIC = calc.optimal_lambda_bic, opt1SE = calc.optimal_lambda_1se;
  const optCVIdx = lambdas.indexOf(optCV), optAICIdx = lambdas.indexOf(optAIC), optBICIdx = lambdas.indexOf(optBIC), opt1SEIdx = lambdas.indexOf(opt1SE);
  // Table
  let html = '';
  html += `<tr><td>Cross-Validation</td><td>${formatNumber(cv[optCVIdx],3)}</td><td>${formatNumber(optCV,3)}</td><td><span style='color:#10B981'>&#10003;</span></td></tr>`;
  html += `<tr><td>AIC (smaller)</td><td>${formatNumber(aic[optAICIdx],1)}</td><td>${formatNumber(optAIC,3)}</td><td><span style='color:#10B981'>&#10003;</span></td></tr>`;
  html += `<tr><td>BIC (smaller)</td><td>${formatNumber(bic[optBICIdx],1)}</td><td>${formatNumber(optBIC,3)}</td><td><span style='color:#10B981'>&#10003;</span></td></tr>`;
  html += `<tr><td>1-SE Rule</td><td>${formatNumber(cv[opt1SEIdx],3)}</td><td>${formatNumber(opt1SE,3)}</td><td><span style='color:#10B981'>&#10003;</span></td></tr>`;
  document.getElementById('optimal-lambda-table-body').innerHTML = html;

  // Recommendation box
  const reco = `<strong>Recommendation:</strong> λ ∈ [${formatNumber(lambdas[1],3)}, ${formatNumber(lambdas[4],3)}]
    <br>Current λ = ${formatNumber(currLambda,3)}
    <br>${currLambda < lambdas[1] ? '<span style="color:#059669;">Current λ is optimal or close to optimal.</span>' : currLambda > lambdas[4] ? '<span style="color:#DC2626;">Current λ is higher than recommended. Consider decreasing for better fit.</span>' : '<span style="color:#FBBF24;">Current λ is in moderate to conservative range.</span>'}`;
  document.getElementById('optimal-lambda-reco-box').innerHTML = reco;

  // Charts
  const colors = ['#1FB8CD', '#FFC185', '#B4413C', '#DB4545', '#5D878F', '#D2BA4C'];
  // CV Chart
  if (chartOptimalCV) chartOptimalCV.destroy();
  chartOptimalCV = new Chart(document.getElementById('chart-optimal-cv').getContext('2d'), {
    type: 'line',
    data: {
      labels: lambdas.map(v => formatNumber(v, 3)),
      datasets: [{
        label: 'CV Error',
        data: cv,
        borderColor: colors[0],
        backgroundColor: colors[0] + '20',
        pointRadius: lambdas.map((v,i)=>(i===lambdaIdx?7: (i===optCVIdx||i===opt1SEIdx?5:3))),
        pointBackgroundColor: lambdas.map((v, i) => i === lambdaIdx ? '#DB4545' : (i===optCVIdx?'#1FB8CD':i===opt1SEIdx?'#D2BA4C':colors[0])),
        borderWidth: 2,
        fill: false
      }]
    },
    options: {
      responsive:true,
      plugins: { legend: {display:false}, tooltip:{} },
      scales: { x:{title: {display:true,text:'λ'}}, y:{title:{display:true, text:'CV Error'}} }
    }
  });
  // AIC Chart
  if (chartOptimalAIC) chartOptimalAIC.destroy();
  chartOptimalAIC = new Chart(document.getElementById('chart-optimal-aic').getContext('2d'), {
    type:'line',
    data:{
      labels: lambdas.map(v => formatNumber(v,3)),
      datasets:[{
        label:'AIC',
        data:aic,
        borderColor: colors[3],
        backgroundColor: colors[3]+'20',
        pointRadius:lambdas.map((v,i)=>(i===lambdaIdx?7: (i===optAICIdx?5:3))),
        pointBackgroundColor: lambdas.map((v,i)=>i===lambdaIdx?'#DB4545':i===optAICIdx?'#DB4545':colors[3]),
        fill:false
      }]
    },
    options:{
      responsive:true,
      plugins:{legend:{display:false},tooltip:{}},
      scales:{x:{title:{display:true, text:'λ'}},y:{title:{display:true,text:'AIC'}}}
    }
  });
  // BIC Chart
  if (chartOptimalBIC) chartOptimalBIC.destroy();
  chartOptimalBIC = new Chart(document.getElementById('chart-optimal-bic').getContext('2d'), {
    type:'line',
    data:{
      labels:lambdas.map(v=>formatNumber(v,3)),
      datasets:[{
        label:'BIC',
        data:bic,
        borderColor: colors[4],
        backgroundColor: colors[4]+'20',
        pointRadius:lambdas.map((v,i)=>(i===lambdaIdx?7: (i===optBICIdx?5:3))),
        pointBackgroundColor: lambdas.map((v,i)=>i===lambdaIdx?'#DB4545':i===optBICIdx?'#5D878F':colors[4]),
        fill:false
      }]
    },
    options:{
      responsive:true,
      plugins:{legend:{display:false},tooltip:{}},
      scales:{x:{title:{display:true, text:'λ'}},y:{title:{display:true,text:'BIC'}}}
    }
  });
}

function updateAllWithOptimal() {
  updateAll();
  updateOptimalLambdaSection();
}

window.addEventListener('load', () => {
  createCharts();
  updateAllWithOptimal();
});

document.getElementById('lambda-slider').addEventListener('input', (e) => {
  const logValue = parseFloat(e.target.value);
  state.lambda = Math.pow(10, logValue);
  document.getElementById('lambda-value').textContent = formatNumber(state.lambda, 3);
  document.getElementById('current-lambda-display').textContent = formatNumber(state.lambda, 3);
  updateAllWithOptimal();
});

// recompute also when method/dataset changes
const originalBtns = [...document.querySelectorAll('.btn-dataset'), ...document.querySelectorAll('.btn-method')];
originalBtns.forEach(btn => {
  btn.addEventListener('click', updateOptimalLambdaSection);
});
