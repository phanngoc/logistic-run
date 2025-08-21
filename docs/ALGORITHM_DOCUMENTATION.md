# Tài Liệu Thuật Toán - Logistics Route Optimization

## Tổng Quan Hệ Thống

Hệ thống Logistics Route Optimization sử dụng kết hợp các thuật toán tối ưu hóa truyền thống và Machine Learning để giải quyết bài toán Vehicle Routing Problem (VRP) với các ràng buộc thực tế.

## 1. Kiến Trúc Thuật Toán

### 1.1 Workflow Chính

```
Input: Orders + Fleet + Constraints
    ↓
[Greedy Initial Solution]
    ↓
[ML Enhancement] → [Route Scoring] → [Route Ranking]
    ↓
[Local Search Optimization]
    ↓
Output: Optimized Routes
```

### 1.2 Các Components Chính

1. **Greedy Solver** - Tạo initial solution
2. **ML Predictor Service** - Predict route performance
3. **ML Enhanced Solver** - Optimize bằng ML insights
4. **Local Search** - Fine-tuning solution

## 2. Thuật Toán Greedy Base

### 2.1 Greedy Insertion Algorithm

**Mục tiêu**: Tạo initial solution khả thi nhanh chóng

**Thuật toán**:
```
1. Sort orders theo priority (time window, distance, etc.)
2. For each order:
   a. Find best truck có capacity và time window phù hợp
   b. Insert order vào route cost-optimal position
   c. Update truck state (capacity, time, location)
3. Handle unassigned orders
```

**Complexity**: O(n²m) với n = orders, m = trucks

**Ưu điểm**:
- Fast execution (< 1s cho 100 orders)
- Guarantee feasible solution
- Good baseline cho optimization

**Nhược điểm**:
- Local optimum
- Không consider global constraints

### 2.2 Route Construction Details

**Cost Function**:
```
route_cost = α * distance_cost + β * time_cost + γ * overtime_cost + δ * delay_penalty
```

Với:
- α, β, γ, δ là weights có thể config
- distance_cost = total_km * cost_per_km
- time_cost = total_hours * cost_per_hour
- overtime_cost = overtime_hours * overtime_rate
- delay_penalty = Σ(late_minutes * penalty_rate)

## 3. Machine Learning Enhancement

### 3.1 ML Models Used

Hệ thống sử dụng 3 ML models chính:

#### A. Route Score Predictor
**Mục tiêu**: Predict chất lượng tổng thể của route

**Features** (10 features):
- `total_distance_km`: Tổng khoảng cách route
- `orders_served`: Số orders trong route
- `overtime_hours`: Giờ làm thêm dự kiến
- `utilization_rate`: Tỷ lệ sử dụng capacity
- `distance_per_order`: Khoảng cách trung bình/order
- `is_weekend`: Route chạy cuối tuần
- `is_peak_hour`: Route trong giờ cao điểm
- `shift_start_hour`: Giờ bắt đầu ca
- `shift_end_hour`: Giờ kết thúc ca
- `overtime_threshold_min`: Ngưỡng overtime

**Target**: Route score (0-1, cao = tốt)

**Model**: Random Forest Regressor
- Accuracy: MAE = 0.098, R² = -0.206
- Training data: 500 samples

#### B. Total Cost Predictor
**Mục tiêu**: Predict tổng chi phí route

**Features**: Same as Route Score
**Target**: Total cost (USD)
**Model**: Random Forest Regressor
- Accuracy: MAE = 92.2, R² = -0.208

#### C. Overtime Hours Predictor
**Mục tiêu**: Predict số giờ overtime

**Features**: Same as Route Score
**Target**: Overtime hours
**Model**: Random Forest Regressor
- Accuracy: MAE = 0.0, R² = 1.0 (Perfect fit)

### 3.2 ML Enhancement Workflow

```
1. Generate initial routes (Greedy)
2. Extract features cho mỗi route
3. Predict performance (score, cost, overtime)
4. Rank routes theo ML predictions
5. Apply ML-guided improvements:
   - Reorder stops cho time efficiency
   - Adjust timing cho port delays
   - Optimize resource allocation
```

### 3.3 Route Ranking Algorithm

```python
def rank_routes(route_predictions):
    # Weighted scoring
    for route, prediction in route_predictions:
        ml_score = (
            0.4 * prediction.predicted_score +
            0.3 * cost_normalization(prediction.predicted_cost) +
            0.2 * overtime_penalty(prediction.predicted_overtime_hours) +
            0.1 * prediction.confidence
        )
    
    # Sort by ml_score descending
    return sorted(route_predictions, key=lambda x: ml_score, reverse=True)
```

## 4. Local Search Optimization

### 4.1 ML-Guided Local Search

**Mục tiêu**: Improve routes dựa trên ML insights

**Strategies**:

#### A. Stop Reordering
- **Trigger**: Predicted overtime > 1.0 hours
- **Method**: Minimize travel time giữa stops
- **Algorithm**: Nearest neighbor với time windows

#### B. Port Delay Adjustment
- **Trigger**: Routes qua ports với high dwell time
- **Method**: Adjust ETA/ETD based on ML port predictions
- **Algorithm**: Propagate delays through route timeline

#### C. Capacity Optimization
- **Trigger**: Low utilization rate (< 70%)
- **Method**: Merge/split routes để improve utilization
- **Algorithm**: 2-opt style moves between routes

### 4.2 Improvement Metrics

Sau mỗi local search iteration:
```
improvement_score = (new_ml_score - old_ml_score) / old_ml_score
```

Chỉ accept improvements nếu improvement_score > threshold (default: 0.02)

## 5. External Data Integration

### 5.1 Travel Time Service

**Purpose**: Real-time travel time estimation

**Data Sources**:
- OpenRouteService API
- Redis cache cho performance
- Historical traffic patterns

**Features**:
- Route optimization với traffic
- Time-dependent routing
- Distance matrix calculation

### 5.2 Port API Service

**Purpose**: Port dwell time prediction

**Data Sources**:
- Port congestion data
- Historical dwell times
- Queue length estimates

**ML Application**:
- Predict gate waiting time
- Optimize port visit timing
- Route scheduling around port delays

## 6. Performance Optimization

### 6.1 Computational Complexity

**Overall Complexity**: O(n²m + k*I)
- n = number of orders
- m = number of trucks  
- k = routes to optimize
- I = local search iterations

**Typical Performance**:
- 50 orders, 10 trucks: ~2-3 seconds
- 100 orders, 20 trucks: ~5-8 seconds
- 200 orders, 40 trucks: ~15-25 seconds

### 6.2 Scalability Considerations

**Memory Usage**:
- O(nm) cho distance matrix
- O(n) cho ML feature vectors
- O(k) cho route storage

**Parallel Processing**:
- ML predictions có thể parallel
- Independent route improvements
- Batch API calls cho external services

## 7. Model Training & Validation

### 7.1 Training Data Generation

**Synthetic Data Approach**:
```python
def generate_training_sample():
    # Random route parameters
    distance = random(50, 500)  # km
    orders = random(3, 15)
    utilization = random(0.3, 1.0)
    
    # Simulate route execution
    route_score = calculate_route_score(...)
    total_cost = simulate_costs(...)
    overtime = calculate_overtime(...)
    
    return features, targets
```

**Data Augmentation**:
- Traffic variations
- Seasonal patterns
- Different truck types
- Port congestion scenarios

### 7.2 Model Validation

**Cross-Validation Strategy**:
- 80/20 train-test split
- 5-fold cross-validation
- Time-series validation cho temporal data

**Metrics Tracking**:
- MAE (Mean Absolute Error)
- R² Score
- Business metrics (cost reduction, on-time delivery)

## 8. Configuration & Tuning

### 8.1 Hyperparameters

**Greedy Solver**:
```yaml
insertion_strategy: "best_position"
time_window_penalty: 100
capacity_penalty: 50
```

**ML Models**:
```yaml
random_forest:
  n_estimators: 100
  max_depth: 10
  min_samples_split: 5
```

**Local Search**:
```yaml
max_iterations: 10
improvement_threshold: 0.02
time_limit_ratio: 0.8
```

### 8.2 Weight Configuration

**Cost Weights**:
- Distance: 0.4 (primary factor)
- Time: 0.3 (efficiency)
- Overtime: 0.2 (cost control)
- Delay: 0.1 (service quality)

**ML Score Weights**:
- Predicted Score: 0.4
- Cost Factor: 0.3
- Overtime Penalty: 0.2
- Confidence: 0.1

## 9. Future Enhancements

### 9.1 Advanced ML Models

**Deep Learning Approaches**:
- Graph Neural Networks cho route structure
- LSTM cho time series prediction
- Transformer models cho sequence optimization

**Reinforcement Learning**:
- Q-Learning cho dynamic routing
- Actor-Critic cho real-time decisions
- Multi-agent systems cho fleet coordination

### 9.2 Real-time Optimization

**Dynamic Re-routing**:
- Traffic incident response
- Customer request changes
- Vehicle breakdown handling

**Continuous Learning**:
- Online model updates
- Feedback integration
- Performance monitoring

## 10. Evaluation & Metrics

### 10.1 Algorithm Performance

**Solution Quality**:
- Cost reduction vs baseline: Target 15-25%
- Service level (on-time delivery): Target 95%+
- Fleet utilization: Target 80%+

**Computational Performance**:
- Response time: < 10s cho 100 orders
- Memory usage: < 1GB
- Scalability: Linear với number of orders

### 10.2 Business Impact

**KPIs**:
- Total transportation cost
- Driver overtime hours
- Customer satisfaction score
- Fuel consumption
- Route compliance rate

**ROI Calculation**:
```
monthly_savings = baseline_cost - optimized_cost
roi = (monthly_savings * 12 - system_cost) / system_cost
```

---

*Tài liệu này được cập nhật định kỳ theo sự phát triển của hệ thống và feedback từ thực tế triển khai.*
