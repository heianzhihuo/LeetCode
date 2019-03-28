import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Stack;

public class LeetCode3 {
	
	public int findKthNumber(int n, int k) {
        int result[] = new int[k];
        int i,x;
        i = 0;
        result[0] = 1;
        x = 1;
        while(i<k){
            while(x<=n && i<k) {
            	System.out.println(i+":"+x);
            	result[i] = x;
                i++;
                x *= 10;
            }
        	x = x/10 + 1;
        }
        return result[k-1];
    }
	
	
	public int sumNum(int n) {
		int g[][]= new int[n+1][n+1];
		for(int i=1;i<n;i++) {
			g[i][1] = 1;
			g[1][i] = 1; 
		}
		for(int i=2;i<=n;i++)
			for(int j=2;j<=i-1;j++) {
				g[i][j] = g[i][j-1];
				if(i>=j*2)
					g[i][j] = g[i][j]+g[i-j][i-j-1]+1;
				else
					g[i][j] = g[i][j]+g[i-j][j];
			}
		return g[n][n-1]+1;
	}
	
	/*45.Jump Game II
	 * 给定一个数组，每个数组表示在当前位置时能一步能跳的最长距离
	 * 求最少跳次数*/
	public int jump(int[] nums) {
        int n = nums.length;
        int i,j;
        if(n<=1)
            return 0;
        int numJ[] = new int[n];
        for(i=0;i<n;i++)
            numJ[i] = Integer.MAX_VALUE;
        numJ[0] = 0;
        for(i=0;i<n-1;i++){
            for(j=1;j<=nums[i] && i+j<n;j++)
                if(numJ[i]+1<numJ[i+j])
                    numJ[i+j] = numJ[i]+1;
        }
        return numJ[n-1];
    }
	public int jump2(int[] nums) {
        int start = 0;
        int end = 0;
        int fatest = 0;
        int count = 0;
        while(end<nums.length-1){
            count++;
            for(int i=start;i<=end;i++)
                if(nums[i]+i>fatest)
                    fatest = nums[i]+i;
            start = end+1;
            end = fatest;
        }
        return count;
    }
	/*55.Jump Game
	 * 判断能不能到达最后的位置*/
	public boolean canJump(int[] nums) {
        int lastPos = nums.length-1;
        for(int i=nums.length-1;i>=0;i--){
            if(i+nums[i]>=lastPos)
                lastPos = i;
        }
        return lastPos==0;
    }
	
	/* 629.K Inverse Pairs Array
	 * n的数组，k个逆序对的排列数
	 * */
	public int kInversePairs(int n, int k) {
        if(n==0)
            return 0;
        if(k==0)
            return 1;
        if(k>n*(n-1)/2)
            return 0;
        int f[][] = new int[n][k+1];
        int i,j,x;
        for(i=0;i<n;i++)
            f[i][0] = 1;
        for(i=1;i<n;i++){
            for(j=1;j<=k;j++){
                for(x=0;x<=i && x<=j;x++)
                    f[i][j] = (f[i][j]+f[i-1][j-x])% 1000000007;
            }
        }
        return f[n-1][k];
    }
	
	public int tryNext(int[][] grid,int rows,int cols,int x,int y){
        if(0<=x && x<rows && 0<=y && y<cols && grid[x][y]!=-1){
            if(x==rows-1 && y==cols-1){
                int i,j;
                int m[][] = new int[rows][cols];
                for(i=0;i<rows;i++)
                	for(j=0;j<cols;j++)
                		m[i][j] = -1;
                m[rows-1][cols-1] = grid[rows-1][cols-1];
                for(i=rows-2;i>=0;i--)
                	if(grid[i][cols-1]!=-1)
                		m[i][cols-1] = m[i+1][cols-1] + grid[i][cols-1];
                	else break;
                for(j=cols-2;j>=0;j--)
                	if(grid[rows-1][j]!=-1)
                		m[rows-1][j] = m[rows-1][j+1]+grid[rows-1][j];
                	else break;
                for(i=rows-2;i>=0;i--)
                	for(j=cols-2;j>=0;j--)
                		if(grid[i][j]!=-1) {
                			if(m[i+1][j]>m[i][j+1] && m[i+1][j]!=-1)
                				m[i][j] = m[i+1][j] + grid[i][j];
                			if(m[i+1][j]<=m[i][j+1] && m[i][j+1]!=-1)
                				m[i][j] = m[i][j+1] + grid[i][j];
                		}
                return m[0][0];
            }else{
	            int a = grid[x][y];
	            int c,d;
	            grid[x][y] = 0;
	            c = tryNext(grid,rows,cols,x+1,y);
	            d = tryNext(grid,rows,cols,x,y+1);
	            grid[x][y] = a;
	            if(c==-1 && d==-1)
	                return -1;
	            if(c>d)
	                return c+a;
	            else
	                return d+a;
            }
        }
        return -1;
    }
    
    public int cherryPickup(int[][] grid) {
        int rows = grid.length;
        int cols = grid[0].length;
        if(rows==1 && cols==1)
            return grid[0][0];
        int res = tryNext(grid,rows,cols,0,0);
        if(res==-1)
        return 0;
        return res;
    }
	
    /* 713. Subarray Product Less Than K
     * 连续数组积小于k的数目
     * 方法一：乘法转对数加法，然后加和
     * 方法二：
     * */
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        int n = nums.length;
        if(n==0 || k<=1)
            return 0;
        int total = 1;
        int i,j,count = 0;
        i = 0;j = 0;
        for(;j<n;j++) {
        	total *= nums[j];
        	while(total>k) {
        		total = total/nums[i];
        		i++;
        	}
        	count += j-i+1;
        }
        /*
        double logk = Math.log(k);
        double sum[] = new double[n+1];
        int i,count = 0,lo,hi;
        for(i=0;i<n;i++)
        	sum[i+1] = sum[i]+Math.log(nums[i]);
        for(i=0;i<n+1;i++) {
        	lo = i+1;
        	hi = n+1;
        	while(lo<hi) {
        		int mi = lo + (hi-lo)/2;
        		if(sum[mi]<sum[i]+logk-1e-9) lo = mi+1;
        		else hi = mi;
        	}
        	count += lo-i-1;
        }*/
        return count;
    }
    
    /*152. Maximum Product Subarray
     * 思想：记录第一个负整数左边的积，最后一个负整数右边的积，以及所有的积
     * */
    public int maxProduct(int[] nums) {
        int n = nums.length;
        if(n==1)
            return nums[0];
        int product=1,left=1,right=1,count = 0;
        int max = 0;
        int i;
        for(i=0;i<n;i++)
            if(nums[i]==0){
                if(count>0 && product>max)
                    max = product;
                else if(product<0 && count>1){
                    if(product/left>max)
                        max = product/left;
                    if(product/right>max)
                        max = product/right;
                }
                product = 1;
                left =1;
                right = 1;
                count = 0;
            }
            else{
                count++;
                product *= nums[i];
                if(left>0)
                    left = product;
                if(nums[i]<0)
                    right = 1;
                right *= nums[i];
            }
        if(count>0 && product>max)
            max = product;
            else if(product<0 && count>1){
                if(product/left>max)
                    max = product/left;
                if(product/right>max)
                    max = product/right;
            }
        return max;
    }
    
    /*978. Longest Turbulent Subarray
     * 最长连续波动子数组
     * 增减增减交替
     * */
    public int maxTurbulenceSize(int[] A) {
        int n = A.length;
        if(n<=2)
            return 1;
        int i,j,max=0;
        int flag = 0;
        i = 0;
        j = 1;
        while(j<n){
            int x = A[j]-A[j-1];
            if(x>0)
                x = 1;
            else if(x<0)
                x = -1;
            if(x==0){
                if(j-i>max)
                    max = j-i;
                i = j;
                flag = 0;
            }else if(flag*x>0){
                if(j-i>max)
                    max = j-i;
                i = j-1;
            }else
                flag = x;
            j++; 
        }
        if(j-i>max)
            max = j-i;
        return max;
    }
    
    /* 576. Out of Boundary Paths
     * 在m*n的网格的位置i,j处，在N步以内出边界的方法数
     * 出去后不能回来
     * */
    public int findPaths(int m, int n, int N, int i, int j) {
        if(N<=0)
            return 0;
        int cur[][] = new int[m][n];
        int x,y,z;
        int count = 0;
        for(x=0;x<m;x++){
            for(y=0;y<n;y++){
                if(x==0)
                    cur[x][y] += 1;
                if(x==m-1)
                    cur[x][y] += 1;
                if(y==0)
                    cur[x][y] += 1;
                if(y==n-1)
                    cur[x][y] += 1;
            }
        }
        count = cur[i][j];
        for(z=1;z<N;z++){
            int next[][] = new int[m][n];
            for(x=0;x<m;x++)
                for(y=0;y<n;y++){
                    if(x>0)
                        next[x][y] = (next[x][y]+cur[x-1][y])%1000000007;
                    if(x<m-1)
                        next[x][y] = (next[x][y]+cur[x+1][y])%1000000007;
                    if(y>0)
                        next[x][y] = (next[x][y]+cur[x][y-1])%1000000007;
                    if(y<n-1)
                        next[x][y] = (next[x][y]+cur[x][y+1])%1000000007;
                }
            count = (count+next[i][j])%1000000007;
            cur = next;
        }
        return count;
    }
    
    /* 688. Knight Probability in Chessboard
     * 在N*N的棋盘上，马在位置r行c列，尝试跳K步
     * 每次随机从8个方向中选择一个方向移动，可能跳出去(出去后不再回来)
     * 求K步后，马仍然在棋盘上的概率
     * 思想：广度优先，递归
     * 
     * */
    public double knightProbability(int N, int K, int r, int c) {
        if(K<=0)
            return 1;
        double cur[][] = new double[N][N];
        cur[r][c] = 1.0;
        int i,j,x;
        for(x=0;x<K;x++){
            double next[][] = new double[N][N];
            for(i=0;i<N;i++)
                for(j=0;j<N;j++)
                    if(cur[i][j]>1e-9){
                        if(i-1>=0 && j-2>=0)
                            next[i-1][j-2] += cur[i][j]/8;
                        if(i-1>=0 && j+2<N)
                            next[i-1][j+2] += cur[i][j]/8;
                        if(i+1<N && j-2>=0)
                            next[i+1][j-2] += cur[i][j]/8;
                        if(i+1<N && j+2<N)
                            next[i+1][j+2] += cur[i][j]/8;
                        if(i-2>=0 && j-1>=0)
                            next[i-2][j-1] += cur[i][j]/8;
                        if(i-2>=0 && j+1<N)
                            next[i-2][j+1] += cur[i][j]/8;
                        if(i+2<N && j-1>=0)
                            next[i+2][j-1] += cur[i][j]/8;
                        if(i+2<N && j+1<N)
                            next[i+2][j+1] += cur[i][j]/8;
                    }
            cur = next;
        }
        double sum = 0;
        for(i=0;i<N;i++)
            for(j=0;j<N;j++)
                sum += cur[i][j];
        return sum;
    }
    
    /*552. Student Attendance Record II
     * A:Absent
     * L:Late
     * P:Present
     * 找出长度为n的好串的数目
     * 好串的定义：至多1个A，至多2个连续的L
     * 思想是：含A，不含A
     * 结尾是L，LL、非L
     * */
    public int checkRecord(int n) {
        if(n==0)
            return 0;
        if(n==1)
            return 3;
        int i,j;
        int cur[] = new int[6];
        //0:不含A，非L；1：不含A，L；2：不含A，LL
        //3:含A，非L；4：含A，L；3：含A，LL
        cur[0] = 1;
        cur[1] = 1;
        cur[3] = 1;
        for(i=1;i<n;i++){
            int next[] = new int[6];
            //+A
            for(j=0;j<3;j++)
                next[3] = (next[3]+cur[j]) % 1000000007;
            //+L
            for(j=0;j<2;j++){
                next[j+1] = (next[j+1]+cur[j]) % 1000000007;
                next[j+4] = (next[j+4]+cur[j+3]) % 1000000007;
            }
            //+P
            for(j=0;j<3;j++){
                next[0] = (next[0]+cur[j])% 1000000007;
                next[3] = (next[3]+cur[3+j])% 1000000007;
            }
            cur = next;   
        }
        int res = 0;
        for(j=0;j<6;j++)
            res = (res+cur[j])% 1000000007;
        return res;
    }
    
    /*140. Word Break II
     * 给定一个字符串s，和一个字典wordDict
     * 把s拆分成若干个单词，每个单词之间用空格隔开，单词必须在wordDict中出现过
     * wordDict中的单词可以重复使用
     * 求所有拆分结果
     * */
    HashMap<String,List<String>> cache = new HashMap<>();
    public List<String> wordBreak(String s, List<String> wordDict) {
        if(cache.containsKey(s))
            return cache.get(s);
        List<String> res = new ArrayList<>();
        List<String> tmp;
        if(s.length()==0)
            return res;
        if(wordDict.contains(s)) res.add(s);
        for(String pre:wordDict){
            if(s.startsWith(pre) && !s.equals(pre)){
                tmp = wordBreak(s.substring(pre.length()),wordDict);
                for(String fo:tmp)
                    res.add(pre+" "+fo);
                    
            }
        }
        cache.put(s,res);
        return res;
    }
    
    /* 139. Word Break
     * 判断给定字符串s能否由给定单词表中的wordDict的单词组成
     * 用于判断每个字符前是否可以用
     * 动态规划
     * */
    public boolean wordBreak1(String s, List<String> wordDict) {
        boolean table[] = new boolean[s.length()+1];
        int i,j;
        table[0] = true;
        for(i=1;i<=s.length();i++){
            for(j=0;j<i;j++){
                table[i] = table[j] && wordDict.contains(s.substring(j,i));
                if(table[i]) break;
            }
        }
        return table[s.length()];
    }
    
    /* 68. Text Justification
     * 讲单词组成行，每行最大长度位maxWidth，
     * 两个单词之间必须有空格，同一行的空格要均匀分布，多的空格优先分配给左边的间隔
     * 最后一行要左对齐，最右必须是空格。
     * */
    public List<String> fullJustify(String[] words, int maxWidth) {
        int i,j,k,t,n = words.length;
        List<String> res = new ArrayList<>();
        int cur = 0,left;
        i = 0;
        while(i<n){
            j = i;
            for(;j<n;j++)
                if(cur<maxWidth)
                    cur += words[j].length()+1;
                else
                    break;
            if(cur>maxWidth+1){
                cur -= words[j].length()+1;
            }else{
                cur--;
                j++;
            }
            left = maxWidth-cur;
            String tmp = "";
            if(j==n){
                for(k=i;k<j-1;k++)
                    tmp += words[k]+" ";
                tmp += words[k];
                for(k=0;k<left;k++)
                    tmp += " ";
                res.add(tmp);
                return res;
            }else{
                for(k=i;k<j-1;k++){
                    tmp += words[k]+" ";
                    if(k-i<left%(j-i-1)){
                        for(t=0;t<left/(j-i-1)+1;t++)
                            tmp += " ";
                    }
                    else{
                        for(t=0;t<left/(j-i-1);t++)
                            tmp += " ";
                    }
                    tmp += words[k];
                    res.add(tmp);
                }
            }
            i = j;
        }
        return res;
        
    }
    
    /*221. Maximal Square
     * 最大正方形
     * 给定一个2D的矩阵，由‘0’和‘1’组成
     * 找到最大的子方阵，使得它的面积最大
     * 动态规划
     * */
    public int maximalSquare(char[][] matrix) {
        int m = matrix.length;
        if(m<=0)
            return 0;
        int n = matrix[0].length;
        int i,j;
        int max[][] = new int[m][n];
        int x,mm = 0;;
        for(i=0;i<m;i++){
            max[i][0] = matrix[i][0]-'0';
            if(max[i][0]>mm)
                mm = max[i][0];
        }
            
        for(j=0;j<n;j++){
            max[0][j] = matrix[0][j]-'0';
            if(max[0][j]>mm)
                mm = max[0][j];
        }
            
        for(i=1;i<m;i++)
            for(j=1;j<n;j++)
                if(matrix[i][j]=='0')
                    max[i][j] = 0;
                else{
                    x = max[i-1][j-1];
                    if(max[i-1][j]<x)
                        x = max[i-1][j];
                    if(max[i][j-1]<x)
                        x = max[i][j-1];
                    if(x>=mm)
                        mm = x+1;
                    max[i][j] = x+1;
                }
        return mm*mm;
    }
    
    /*85. Maximal Rectangle
     * 最大矩形
     * 给定一个2D矩阵，由‘0’和‘1’组成
     * 找到面积最大的矩形，由1组成
     * 方法，到第i行的高度算出来，然后计算这个高度的最大面积
     * */
    public int maxA(int[] A){
        int n = A.length;
        if(n<=0) return 0;
        Stack<Integer> stack = new Stack<>();
        int i,h,s;
        int maxArea = 0;
        for(i=0;i<=n;i++){
            if(i<n)
                h = A[i];
            else
                h = 0;
            if(stack.isEmpty() || h>=A[stack.peek()])
                stack.push(i);
            else{
                int tp = stack.pop();
                if(stack.isEmpty())
                    s = A[tp] * i;
                else
                    s = A[tp] * (i-1-stack.peek());
                if(s>maxArea)
                    maxArea = s;
                i--;
            }
                
        }
        return maxArea;
    }
    public int maximalRectangle(char[][] matrix) {
        int m = matrix.length;
        if(m<=0)
            return 0;
        int n = matrix[0].length;
        int t[] = new int[n];
        int maxArea = 0,x,i,j;
        x = 0;
        for(j=0;j<n;j++){
            t[j] = matrix[0][j]-'0';
        }
        maxArea = maxA(t);
        for(i=1;i<m;i++){
            for(j=0;j<n;j++)
                if(matrix[i][j]=='0')
                    t[j] = 0;
                else
                    t[j]++;
            x = maxA(t);
            if(x>maxArea)
                maxArea = x;
        }
        return maxArea;
    }
    
    /* 84. Largest Rectangle in Histogram
     * 给定一个n个非负数组，求直方图中最大的矩形
     * 方法：从左往右扫描，找到左边第一个比当前位置小的数的位置
     * 从右往左扫描，找到右边第一个比当前位置小的数的位置
     * */
    public int largestRectangleArea(int[] heights) {
        int i,j;
        int n = heights.length;
        if(n==0)
            return 0;
        if(n==1)
            return heights[0];
        int left[] = new int[n];
        int right[] = new int[n];
        left[0] = -1;
        for(i=1;i<n;i++)
            if(heights[i]>heights[i-1])
                left[i] = i-1;
            else if(heights[i]==heights[i-1])
                left[i] = left[i-1];
            else{
                j = left[i-1];
                while(j!=-1 && heights[i]<=heights[j]) j = left[j];
                left[i] = j;                
            }
        right[n-1] = n;
        for(i=n-2;i>=0;i--)
            if(heights[i]>heights[i+1])
                right[i] = i+1;
            else if(heights[i]==heights[i+1])
                right[i] = right[i+1];
            else{
                j = right[i+1];
                while(j!=n && heights[i]<=heights[j]) j = right[j];
                right[i] = j;
            }
        int maxArea = 0;
        for(i=0;i<n;i++)
            maxArea = Math.max(maxArea,heights[i]*(right[i]-left[i]-1));
        return maxArea;
    }
    
    /* 228. Summary Ranges
     * 把有序数组转换成范围形式
     * 即：连续的整数转换成最小->最大
     * */
    public List<String> summaryRanges(int[] nums) {
        List<String> res = new ArrayList<>();
        int n = nums.length;
        String tmp;
        if(n==0)
            return res;
        if(n==1){
            tmp = nums[0]+"";
            res.add(tmp);
            return res;
        }
        int i,j;
        i = 0;
        while(i<n){
            j = i+1;
            while(j!=n && nums[j]==nums[j-1]+1) j++;
            tmp = ""+nums[j-1];
            if(j-i>1)
                tmp = nums[i]+"->"+tmp;
            res.add(tmp);
            i = j;
        }
        return res;
    }
    
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		LeetCode3 test = new LeetCode3();
		String tmpsss = 1+"";
		System.out.println(tmpsss);
		
		
		int heights[] = {0,1,0,2,1,0,1,3,2,1,2,1};
			//{1,2,2};
		System.out.println(test.largestRectangleArea(heights));
		
		char matrixxxx[][] = {
			{'1','0','1','0','0'},
			{'1','0','1','1','1'},
			{'1','1','1','1','1'},
			{'1','0','0','1','0'}};
		System.out.println(test.maximalRectangle(matrixxxx));
		
		
		String words[] = {"This", "is", "an", "example", "of", "text", 
				"justification."};
		List<String> resss = test.fullJustify(words, 16);
		for(String ss:resss) 
			System.out.println(ss);
		
		
		List<String> wordDict = new ArrayList<>();
		wordDict.add("leet");
		wordDict.add("code");
		System.out.println(test.wordBreak1("leetcode", wordDict));
		
		System.out.println(test.knightProbability(3, 2, 0, 0));
		
		int B[] = {0,-2,0};
		
		System.out.println(test.maxProduct(B));
		
		String str = "aaaa";
		int A[] = {10,9,10,4,3,8,3,3,6,2,10,10,9,3};
		System.out.println(test.numSubarrayProductLessThanK(A, 19));
				
		
		
		
		/*
		 * {{1,1,1,1,0,0,0},
				{0,0,0,1,0,0,0},
				{0,0,0,1,0,0,1},
				{1,0,0,1,0,0,0},
				{0,0,0,1,0,0,0},
				{0,0,0,1,0,0,0},
				{0,0,0,1,1,1,1}};*/
		/*
		 * [
		 * [1,1,1,1,-1,-1,-1,1,0,0],
		 * [1,0,0,0,1,0,0,0,1,0],
		 * [0,0,1,1,1,1,0,1,1,1],
		 * [1,1,0,1,1,1,0,-1,1,1],
		 * [0,0,0,0,1,-1,0,0,1,-1],
		 * [1,0,1,1,1,0,0,-1,1,0],
		 * [1,1,0,1,0,0,1,0,1,-1],
		 * [1,-1,0,1,0,0,0,1,-1,1],
		 * [1,0,-1,0,-1,0,0,1,0,0],
		 * [0,0,-1,0,1,0,1,0,0,1]]*/
		int [][] grid = {{1,1,1,1,-1,-1,-1,1,0,0},
				{1,0,0,0,1,0,0,0,1,0},
				{0,0,1,1,1,1,0,1,1,1},
				{1,1,0,1,1,1,0,-1,1,1},
				{0,0,0,0,1,-1,0,0,1,-1},
				{1,0,1,1,1,0,0,-1,1,0},
				{1,1,0,1,0,0,1,0,1,-1},
				{1,-1,0,1,0,0,0,1,-1,1},
				{1,0,-1,0,-1,0,0,1,0,0},
				{0,0,-1,0,1,0,1,0,0,1},
				
		};
		System.out.println(test.cherryPickup(grid));
		
		System.out.println(test.kInversePairs(3, 1));
		
		//System.out.println(test.findKthNumber(100, 90));
		for(int i=1;i<10;i++) {
			System.out.println(i+":"+test.sumNum(i));
		}
	}

}
