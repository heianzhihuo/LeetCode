# BinarySearch
二分查找的各种变形和应用

# 原始的二分查找
```java
int search(int[] A,int x){
	int i = 0,j = A.length-1;
	//这里是重点，必须是<=
	while(i<=j){
		int mid = (i+j)/2;
		if(A[mid]==x)
			return mid;
		else if(A[mid]<x])
			i = mid + 1;
		else
			j = mid - 1;
	}
	return -1;
}
```
# 二分查找变种
## 2.1查找第一个与key相等的元素
```java
int findFirstEqual(int[] A,int x){
	int i = 0,j = A.length-1;
	while(i<=j){
		int mid = (i+j)/2;
		if(A[mid]>=x)
			j = mid - 1;
		else
			i = mid + 1;
	}
	if(i<A.length && A[i]==x)
		return i;
	return -1;
}
```

## 2.2查找最后一个与key相等的元素
```java
int findLastEqual(int[] A,int x){
	int i = 0,j = A.length-1;
	while(i<=j){
		int mid = (i+j)/2;
		if(A[mid]<=x)
			i = mid + 1;
		else
			j = mid - 1;
	}
	if(j>=0 && A[j]==x)
		return j;
	return -1;
}
```
