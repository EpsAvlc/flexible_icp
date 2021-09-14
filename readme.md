# Flexible ICP

An implementation of ICP(Iterative Closest Point). Users can specify which dimensions of transformation to optimize.

## Progress
- [x] standarad icp
- [x] point-to-plane icp
- [ ] generalized icp

## Usage
see app/flexible_icp_test.cpp

### example 1

```cpp
// method: point-to-plane icp
// optimize: only x,y,z will be optimized in transformtaion,
//   and r p y will not be optimized.
pcl::FlexibleICP ficp(pcl::FlexibleICP::POINT_TO_PLANE, "111000"); 
ficp.setInputSource(source_cloud);
ficp.setInputTarget(target_cloud);
ficp.setMaxCorrespondenceDistance(5);
pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
ficp.align(aligned_cloud);
```

output transformation: 
```
1         0         0 -0.451169
0         1         0 -0.110301
0         0         1 0.0387694
0         0         0         1
```

### example 2
```cpp
// method: point-to-plane icp
// optimize:  x,y,z will not be optimized,
//   and r p y will be optimized.
pcl::FlexibleICP ficp(pcl::FlexibleICP::POINT_TO_PLANE, "000111"); 
ficp.setInputSource(source_cloud);
ficp.setInputTarget(target_cloud);
ficp.setMaxCorrespondenceDistance(5);
pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
ficp.align(aligned_cloud);
```

output transformation:
```
    0.999997   0.00211456  0.000858083            0
 -0.00211745     0.999992    0.0033873            0
-0.000850913   -0.0033891     0.999994            0
           0            0            0            1
```
