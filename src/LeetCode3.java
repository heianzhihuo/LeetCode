import java.util.*;

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
    
    class Node {
        public int val;
        public Node prev;
        public Node next;
        public Node child;
        
        public Node() {}

        public Node(int _val,Node _prev,Node _next,Node _child) {
            val = _val;
            prev = _prev;
            next = _next;
            child = _child;
        }
    };
    /* 430. Flatten a Multilevel Doubly Linked List
     * 把多层的链表转换成单层链表
     * Flatten the list so that all the nodes appear in a single-level,
     *  doubly linked list. 
     *  You are given the head of the first level of the list.
     * */
    public Node flatten(Node head) {
        if(head==null)
            return null;
        Node ch = flatten(head.child);
        flatten(head.next);
        if(ch!=null){
            while(ch.next!=null) ch = ch.next;
            ch.next = head.next;
            if(head.next!=null)
                head.next.prev = ch;
            head.next = head.child;
            head.next.prev = head;
            head.child = null;
        }
        return head;
    }
    
    public class TreeNode{
    	int val;
    	TreeNode left,right;
    	public TreeNode(int val) {
    		this.val = val;
    		left = null;
    		right = null;
    	}
    }
    /* 98. Validate Binary Search Tree
     * 判断一棵二叉树是否是搜索二叉树
     * */
    TreeNode prev = null;
    public boolean isValidBST(TreeNode root) {
        if(root==null)
            return true;
        if(isValidBST(root.left)){
            boolean flag = true;
            if(prev!=null && prev.val>=root.val)
                flag = false;
            prev = root;
            return flag && isValidBST(root.right);
        }
        return false;
    }
    
    /* 236. Lowest Common Ancestor of a Binary Tree
     * 最近公共祖先算法
     * 递归*/
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root==null)
            return null;
        if(root==p)
            return root;
        if(root==q)
            return root;
        TreeNode left = lowestCommonAncestor(root.left,p,q);
        TreeNode right = lowestCommonAncestor(root.right,p,q);
        if(left==null && right==null)
            return null;
        if(left!=null && right!=null)
            return root;
        if(left!=null)
            return left;
        return right;
    }
    
    /* 235. Lowest Common Ancestor of a Binary Search Tree
     * 二叉搜索树的最近公共祖先
     * */
    public TreeNode lowestCommonAncestor1(TreeNode root, TreeNode p, TreeNode q) {
        if(root==null)
            return null;
        if(p==null)
            return q;
        if(q==null)
            return p;
        if(root.val>p.val && root.val>q.val)
            return lowestCommonAncestor(root.left,p,q);
        if(root.val<p.val && root.val<q.val)
            return lowestCommonAncestor(root.right,p,q);
        return root;
    }
    
    class BSTIterator {
        
        Stack<TreeNode> stack ;
        public BSTIterator(TreeNode root) {
            stack = new Stack<>();
            while(root!=null){
                stack.push(root);
                root = root.left;
            }
        }
        /** @return the next smallest number */
        public int next() {
            TreeNode cur = stack.pop();
            if(cur.right!=null){
                TreeNode tmp = cur.right;
                while(tmp!=null){
                    stack.push(tmp);
                    tmp = tmp.left;
                }
            }
            return cur.val;
        }
        /** @return whether we have a next smallest number */
        public boolean hasNext() {
            return !stack.isEmpty();
        }
    }
    
    /*124. Binary Tree Maximum Path Sum
     * 求二叉树中和最长的路径*/
    int max = Integer.MIN_VALUE;
    public int sum(TreeNode root){
        if(root==null)
            return 0;
        if(root.val>max)
            max = root.val;
        int left,right;
        left = sum(root.left);
        right = sum(root.right);
        if(left<=0)
            left = 0;
        if(right<=0)
            right = 0;
        if(left+right+root.val>max)
            max = left+right+root.val;
        if(left>right)
            return left+root.val;
        return right+root.val;
    }
    public int maxPathSum(TreeNode root) {
        sum(root);
        return max;
    }
    
    /* 129. Sum Root to Leaf Numbers
     * 求二叉树由根节点到所有叶子节点组成的数的和
     * 树节点的值在0-9范围内
     * 根节点到叶子节点的路径上的数字组成了一个整数
     * */
    int sum = 0;
    public void sumDown(TreeNode root,int p){
        if(root==null)
            return;
        if(root.left==null && root.right==null){
            sum += (p*10+root.val);
            return;
        }
        if(root.left!=null)
            sumDown(root.left,p*10+root.val);
        if(root.right!=null)
            sumDown(root.right,p*10+root.val);
    }
    public int sumNumbers(TreeNode root) {
        sumDown(root,0);
        return sum;
    }
    
    /* 988. Smallest String Starting From Leaf
     * 求由叶子节点到根节点的字符组成的字符串，字典序最小的那个
     * */
    String min_str = null;
    public void smallDown(TreeNode root,String p){
        if(root==null)
            return;
        p = (char)(root.val+'a')+p;
        if(root.left==null && root.right==null){
            if(min_str==null || p.compareTo(min_str)<0)
                min_str = p;
            return;
        }
        if(root.left!=null)
            smallDown(root.left,p);
        if(root.right!=null)
            smallDown(root.right,p);
    }
    
    public String smallestFromLeaf(TreeNode root) {
        smallDown(root,"");
        return min_str;
    }
    
    /* 257. Binary Tree Paths
     * 求二叉树根节点到所有叶子节点的路径
     * */
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> res = new ArrayList<>();
        if(root==null)
            return res;
        List<String> left = binaryTreePaths(root.left);
        List<String> right = binaryTreePaths(root.right);
        for(String x:left)
            res.add(root.val+"->"+x);
        for(String x:right)
            res.add(root.val+"->"+x);
        if(res.size()==0)
            res.add(root.val+"");
        return res;
    }
    
    /* 113. Path Sum II
     * 求二叉树从根节点到叶子节点的路径和等于sum的所有路径
     * */
    List<List<Integer>> result;
    public void pathSumDown(TreeNode root,int sum,List<Integer> tmp){
        if(root==null) return;
        if(root.val==sum && root.left==null && root.right==null){
            tmp.add(root.val);
            result.add(new ArrayList<>(tmp));
            return;
        }
        tmp.add(root.val);
        if(root.left!=null){
            pathSumDown(root.left,sum-root.val,tmp);
            tmp.remove(tmp.size()-1);
        }
        if(root.right!=null){
            pathSumDown(root.right,sum-root.val,tmp);
            tmp.remove(tmp.size()-1);
        }
    }
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        result = new ArrayList<>();
        List<Integer> tmp = new ArrayList<>();
        pathSumDown(root,sum,tmp);
        return result;
    }
    
    /*437. Path Sum III
     * 求所有和等于sum的路径数
     * 路径可以不从根节点和叶节点出发
     * 必须从上到下*/
    public int pathSumFrom(TreeNode root,int sum){
        if(root==null)
            return 0;
        int res = 0;
        if(root.val==sum)
            res++;
        sum -= root.val;
        res += pathSumFrom(root.left,sum);
        res += pathSumFrom(root.right,sum);
        return res;
    }
    public int pathSum1(TreeNode root, int sum) {
        if(root==null)
            return 0;
        return pathSumFrom(root,sum) +  pathSum1(root.left,sum) +  pathSum1(root.right,sum);
    }
    
    /*687. Longest Univalue Path
     * 最长连续路径长度，连续路径上所有的节点值相同
     * */
    int maxLength = 0;
    public int longestpathDown(TreeNode root){
        if(root==null)
            return 0;
        int res = 0;
        int left = longestpathDown(root.left);
        int right = longestpathDown(root.right);
        if(root.left!=null && root.right!=null && root.val==root.left.val && root.val==root.right.val){
            if((left+right+1)>maxLength)
                maxLength = left+right+1;
        }
        if(root.left!=null && root.val==root.left.val){
            if(left>res)
                res = left;
        }
        if(root.right!=null && root.val==root.right.val){
            if(right>res)
                res = right;
        }
        if(res+1>maxLength)
            maxLength = res+1;
        return res+1;
    }
    public int longestUnivaluePath(TreeNode root) {
        if(root==null)
            return 0;
        longestpathDown(root);
        return maxLength-1;
    }
    
    /* 
     * 
     * */
    public boolean reachingPoints(int sx, int sy, int tx, int ty) {
    	int t;
        if(sy<sx){
            t = sy;
            sy = sx;
            sx = t;
        }
        if(ty<tx){
            t = ty;
            ty = tx;
            tx = t;
        }
        while(tx>=sx && ty>=sy){
            if(sx==tx && sy==ty)
               return true;
            if(ty%tx==0)
                t = tx;
            else
                t = ty%tx;
            ty = tx;
            tx = t;
        }
        return (sx==tx && sy==ty);
    }
    
    /* 501. Find Mode in Binary Search Tree
     * 找到搜索二叉树中出现次数最多的数
     * */
    int cur = -1,cur_num = 0;
    int freq = -1;
    List<Integer> modes;
    public void search(TreeNode root){
        if(root!=null){
            search(root.left);
            if(cur==root.val)
                cur_num++;
            else{
                if(cur_num>freq){
                    freq = cur_num;
                    modes = new ArrayList<>();
                    modes.add(cur);
                }else if(cur_num==freq)
                    modes.add(cur);
                cur = root.val;
                cur_num = 1;
            }
            search(root.right);
        }
    }
    public int[] findMode(TreeNode root) {
        int res[];
        if(root==null)
            return new int[0];
        search(root);
        if(cur_num>freq){
            freq = cur_num;
            modes = new ArrayList<>();
            modes.add(cur);
        }else if(cur_num==freq)
            modes.add(cur);
        res = new int[modes.size()];
        int i;
        for(i=0;i<modes.size();i++)
            res[i] = modes.get(i);
        return res;
    }
    
    class Node1 {
    	public boolean val;
        public boolean isLeaf;
        public Node1 topLeft;
        public Node1 topRight;
        public Node1 bottomLeft;
        public Node1 bottomRight;

        public Node1() {}

        public Node1(boolean _val,boolean _isLeaf,Node1 _topLeft,Node1 _topRight,Node1 _bottomLeft,Node1 _bottomRight) {
            val = _val;
            isLeaf = _isLeaf;
            topLeft = _topLeft;
            topRight = _topRight;
            bottomLeft = _bottomLeft;
            bottomRight = _bottomRight;
        }
    };
    
    /*558. Quad Tree Intersection
     * 四分树的并*/
    public Node1 intersect(Node1 quadTree1, Node1 quadTree2) {
        if(quadTree1.isLeaf && quadTree2.isLeaf)
            return new Node1(quadTree1.val || quadTree2.val,true,null,null,null,null);
        else if(!quadTree1.isLeaf && !quadTree2.isLeaf){
            Node1 _topLeft = intersect(quadTree1.topLeft,quadTree2.topLeft);
            Node1 _topRight = intersect(quadTree1.topRight,quadTree2.topRight);
            Node1 _bottomLeft = intersect(quadTree1.bottomLeft,quadTree2.bottomLeft);
            Node1 _bottomRight = intersect(quadTree1.bottomRight,quadTree2.bottomRight);
            if(_topLeft.isLeaf && _topRight.isLeaf && _bottomLeft.isLeaf && _bottomRight.isLeaf &&
              _topLeft.val==  _topRight.val && _topRight.val==_bottomLeft.val && _bottomLeft.val==_bottomRight.val)
                return new Node1(_topLeft.val,true,null,null,null,null);
            else
                return new Node1(quadTree1.val || quadTree2.val,false,_topLeft,_topRight,_bottomLeft,_bottomRight);
        }else if(quadTree1.isLeaf){
            if(quadTree1.val==true)
                return new Node1(true,true,null,null,null,null);
            else{
                Node1 _topLeft = intersect(quadTree1,quadTree2.topLeft);
                Node1 _topRight = intersect(quadTree1,quadTree2.topRight);
                Node1 _bottomLeft = intersect(quadTree1,quadTree2.bottomLeft);
                Node1 _bottomRight = intersect(quadTree1,quadTree2.bottomRight);
                return new Node1(quadTree2.val,false,_topLeft,_topRight,_bottomLeft,_bottomRight);
            }
        }else{
            if(quadTree2.val==true)
                return new Node1(true,true,null,null,null,null);
            else{
                Node1 _topLeft = intersect(quadTree1.topLeft,quadTree2);
                Node1 _topRight = intersect(quadTree1.topRight,quadTree2);
                Node1 _bottomLeft = intersect(quadTree1.bottomLeft,quadTree2);
                Node1 _bottomRight = intersect(quadTree1.bottomRight,quadTree2);
                return new Node1(quadTree1.val,false,_topLeft,_topRight,_bottomLeft,_bottomRight);
            }
        }
    }
	
    /* 77.Combinations
     * 求1-n的k组合
     * */
    public void combineNext(List<List<Integer>> res,List<Integer> path,int i,int k,int n){
        if(path.size()==k){
            res.add(new ArrayList<Integer>(path));
            return;
        }
        for(;i<=n;i++){
            path.add(i);
            combineNext(res,path,i+1,k,n);
            path.remove(path.size()-1);
        }
    }
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        combineNext(res,new ArrayList<>(),1,k,n);
        return res;
    }
    
    /*72. Edit Distance
     * 字符串的编辑距离
     * 插入、删除、替换
     * edit[i][j] = min
     * edit[i-1][j]+1 if i>0
     * edit[i][j-1]+1 if j>0
     * edit[i-1][j-1] if i>0 && j>0 && word1[i-1]==word2[j-1]
     * edit[i-1][j-1]+1 if i>0 && j>0 && word1[i-1]!=word2[j-1]
     * */
    public int minDistance(String word1, String word2) {
        int n1 = word1.length(),n2 = word2.length();
        if(n1==0 || n2==0)
            return n2+n1;
        int iv[] = new int[n2+1];
        int i,j;
        for(i=1;i<=n2;i++)
            iv[i] = i;
        for(i=1;i<=n1;i++){
            int tmp = iv[0];//i-1,j-1
            iv[0] = i;
            for(j=1;j<=n2;j++){
                int tmp2 = iv[j];//i-1,j-1
                //iv[j] edit[i-1][j]
                //iv[j-1] edit[i][j-1]
                //tmp 	edit[i-1][j-1]
                if(word1.charAt(i-1)==word2.charAt(j-1))
                    iv[j] = tmp;
                else
                    iv[j] = Math.min(tmp,Math.min(iv[j],iv[j-1]))+1;
                tmp = tmp2;
            }
        }
        return iv[n2];
    }
    /*583. Delete Operation for Two Strings
     * 只有删除操作的最短编辑距离*/
    public int minDistance1(String word1, String word2) {
        int n1 = word1.length();
        int n2 = word2.length();
        if(n1==0 || n2==0)
            return n1+n2;
        int i,j;
        int iv[] = new int[n2+1];
        for(i=1;i<=n2;i++)
            iv[i] = i;
        for(i=1;i<=n1;i++){
            int tmp = iv[0];
            iv[0] = i;
            for(j=1;j<=n2;j++){
                int tmp2 = iv[j];
                if(word1.charAt(i-1)==word2.charAt(j-1))
                    iv[j] = tmp;
                else
                    iv[j] = Math.min(iv[j],iv[j-1])+1;
                tmp = tmp2;
            }
        }
        return iv[n2];
    }
    
    public String simplifyPath(String path) {
        String pathArray[] = path.split("/");
        int n = pathArray.length;
        String res = "";
        int i = 0;
        for(i=0;i<n;i++) {
        	if(pathArray[i].length()>0) {
        		if(pathArray[i].equals(".."))
                    res = "";
                else if(!pathArray[i].equals("."))
                    res += "/"+pathArray[i];
        	}
        }
            
        if(res=="")
            return "/";
        return res;
    }
    
    public class ListNode{
    	int val;
    	ListNode next;
    	public ListNode(int val) {
    		this.val = val;
    		next = null;
    	}
    }
    
    /*23. Merge k Sorted Lists
     * K路链表啊归并*/
    public ListNode mergeKLists(ListNode[] lists) {
        Comparator<ListNode> comparator = new Comparator<ListNode>(){
            @Override
            public int compare(ListNode p1,ListNode p2){
                return p1.val-p2.val;
            }
        };
        Queue<ListNode> listQueue = new PriorityQueue<ListNode>(comparator);
        int n = lists.length;
        if(n==0)
            return null;
        int i;
        for(i=0;i<n;i++)
            if(lists[i]!=null)
                listQueue.add(lists[i]);
        ListNode head = new ListNode(-1);
        ListNode tail = head;
        while(!listQueue.isEmpty()){
            ListNode tmp = listQueue.poll();
            if(tmp.next!=null)
                listQueue.add(tmp.next);
            tail.next = tmp;
            tail = tail.next;
        }
        tail.next = null;
        return head.next;
    }
    /*21. Merge Two Sorted Lists
     * 二路链表归并*/
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode head = new ListNode(-1);
        ListNode tail = head;
        while(l1!=null || l2!=null){
            ListNode tmp;
            if(l1==null || (l1!=null && l2!=null && l2.val<=l1.val)){
                tmp = l2;
                l2 = l2.next;
            }else{
                tmp = l1;
                l1 = l1.next;
            }
            tail.next = tmp;
            tail = tail.next;
        }
        tail.next = null;
        return head.next; 
    }
    /*148. Sort List
     * 归并排序，O(nlogn)对链表进行排序
     * 首先找到最中间的节点，然后分成两半递归的排序
     */
    public ListNode sortList(ListNode head) {
        if(head==null || head.next==null)
            return head;
        ListNode slow = head,fast = head.next.next;
        while(fast!=null && fast.next!=null){
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode l1 = head;
        ListNode l2 = slow.next;
        slow.next = null;
        l1 = sortList(l1);
        l2 = sortList(l2);
        ListNode phead = new ListNode(-1);
        ListNode tail = phead;
        ListNode tmp;
        while(l1!=null || l2!=null){
            if(l1==null || (l1!=null && l2!=null && l2.val<l1.val)){
                tmp = l2;
                l2 = l2.next;
            }else{
                tmp = l1;
                l1 = l1.next;
            }
            tail.next = tmp;
            tail = tail.next;
        }
        tail.next = null;
        return phead.next;
    }
    /*147. Insertion Sort List
     * 链表的插入排序，用一个尾指针表示当前已排序*/
    public ListNode insertionSortList(ListNode head) {
        ListNode phead = new ListNode(-1);
        phead.next = head;
        ListNode tail = head;
        while(tail!=null && tail.next!=null){
            if(tail.next.val>=tail.val){
                tail = tail.next;
                continue;
            }
            ListNode tmp = tail.next;
            tail.next = tmp.next;
            ListNode p = phead;
            while(p!=tail && p.next.val<tmp.val) p = p.next;
            tmp.next = p.next;
            p.next = tmp;
        }
        return phead.next;
    }
    /*263. Ugly Number
     * 判断一个数是否为丑数*/
    public boolean isUgly(int num) {
        if(num==0)
            return false;
        while(num%2==0) num/=2;
        while(num%3==0) num/=3;
        while(num%5==0) num/=5;
        return num==1;
    }
    /*878. Nth Magical Number
     * 第N个魔幻数，第N个既能被A整除页能被B整除的正整数*/
    public int nthMagicalNumber(int N, int A, int B) {
        int mod = 1000000000+7;
        int a,b,i;
        if(A>B){
            a = A;
            b = B;
        }else{
            a = B;
            b = A;
        }
        while(a%b!=0){
            i = a%b;
            a = b;
            b = i;
        }
        a = A/b*B;
        b = a/A + a/B -1;
        int x = N/b,y = N%b;
        long res = (long)x*a%mod;
        if(y==0)
            return (int)res;
        int h[] = {A,B};
        for(i=0;i<y-1;i++){
            if(h[0]>h[1])
                h[1] += B;
            else
                h[0] += A;
        }
        res += Math.min(h[0],h[1]);
        return (int)(res%mod);
    }
    
    /*996. Number of Squareful Arrays
     * 给定一个数组，求数组的全排列数，
     * 使得全排列能满足相邻两个元素之和为完全平方数*/
    HashMap<Integer,Integer> table;
    HashMap<Integer,HashSet<Integer>> next;
    int res = 0;
    public int numSquarefulPerms(int[] A) {
        int n = A.length;
        if(n<=1)
            return n;
        int i;
        HashSet<Integer> tmp;
        table = new HashMap<>();
        next = new HashMap<>();
        for(i=0;i<n;i++){
            if(table.containsKey(A[i]))
                table.put(A[i],table.get(A[i])+1);
            else
                table.put(A[i],1);
            next.put(A[i],new HashSet<>());
        }
            
        for(int x:table.keySet())
            for(int y:table.keySet()){
                int s = (int)Math.sqrt(x+y);
                if(s*s==x+y){
                    tmp = next.get(x);
                    tmp.add(y);
                }
            }
        for(int x:table.keySet())
            dfsSquare(x,A.length-1);
        return res;
    }
    public void dfsSquare(int x,int left){
        table.put(x,table.get(x)-1);
        if(left==0) res++;
        for(int y:next.get(x))
            if(table.get(y)>0)
                dfsSquare(y,left-1);
        table.put(x,table.get(x)+1);
    }
    /*3. Longest Substring Without Repeating Characters
     * 最长不含重复字符的连续子串*/
    public int lengthOfLongestSubstring(String s) {
        if(s.length()==0)
        	return 0;
		int max = 0;
        HashSet<Character> chset = new HashSet<>();
        int i,j;
        i = 0;
        j  = 0;
        while(j<s.length()){
            if(!chset.contains(s.charAt(j))){
                chset.add(s.charAt(j));
            }else{
                if(j-i>max)
                    max = j-i;
                while(i<j && s.charAt(j)!=s.charAt(i)){
                    chset.remove(s.charAt(i));
                    i++;
                }
                i++;
            }
            j++;
        }
        if(j-i>max)
            max = j-i;
		return max;
    }
    /*992. Subarrays with K Different Integers
     * 有K个不同字符的连续子串数目*/
    public int subarraysWithKDistinct(int[] A, int K) {
        int n = A.length;
        if(n==0 || K==0)
            return 0;
        if(K==1)
            return n;
        HashMap<Integer,Integer> table = new HashMap<>();
        int i,j,difnum = 0,cnt = 0;
        i = 0;
        j = 0;
        while(j<n){
            table.put(A[j],table.getOrDefault(A[j],0)+1);
            if(table.get(A[j])==1)
                difnum++;
            if(difnum==K)
                cnt++;
            System.out.println(i+","+j+" "+cnt);
            while(difnum==K+1){
                table.put(A[i],table.get(A[i])-1);
                cnt++;
                
                if(table.get(A[i])==0)
                    difnum--;
                i++;
                System.out.println(i+","+j+" "+cnt);
            }
            j++;
        }
        return cnt;
    }
    /*10. Regular Expression Matching
     * 正则表达式匹配
     * *表示前面字符重复0或者多次
     * .表示任意一个字符*/
    public boolean isMatch(String s, String p) {
        return isMatch(s,0,p,0);
    }
    public boolean isMatch(String s,int i,String p,int j){
        if(j==p.length())
            return i==s.length();
        int k;
        if(i==s.length()){
            for(k=j;k<p.length();k++) 
                if(p.charAt(k)!='*' && (k+1==p.length() || p.charAt(k+1)!='*'))
                    return false;
            return true;
        }
        if(j+1==p.length()){
            //当前匹配位是最后一位
            if(s.charAt(i)==p.charAt(j) || p.charAt(j)=='.')
                return isMatch(s,i+1,p,j+1);
            return false;
        }
        if((s.charAt(i)==p.charAt(j) || p.charAt(j)=='.') && p.charAt(j+1)!='*')
           return isMatch(s,i+1,p,j+1);
        if((s.charAt(i)==p.charAt(j) || p.charAt(j)=='.') && p.charAt(j+1)=='*')
           return isMatch(s,i+1,p,j) || isMatch(s,i,p,j+2);
        if(p.charAt(j+1)=='*')
            return isMatch(s,i,p,j+2);
        return false;
    }
    /*动态规划求解*/
    public boolean isMatch1(String s, String p) {
        int m = s.length();
        int n = p.length();
        int i,j;
        boolean match[][] = new boolean[m+1][n+1];
        match[0][0] = true;
        for(j=1;j<=n;j++)
            if(p.charAt(j-1)!='*' && (j==n || p.charAt(j)!='*'))
                break;
            else
                match[0][j] = true;
        for(i=1;i<=m;i++)
            for(j=1;j<=n;j++){
                if(s.charAt(i-1)==p.charAt(j-1) || p.charAt(j-1)=='.')
                    match[i][j] = match[i-1][j-1];
                else if(p.charAt(j-1)=='*'){
                    match[i][j] = match[i][j-1];
                    if(j>1)
                        match[i][j] = match[i][j] || match[i][j-2];
                    if(j>1 && s.charAt(i-1)==p.charAt(j-2) || p.charAt(j-2)=='.')
                        match[i][j] = match[i][j] || match[i-1][j];
                }
            }
        return match[m][n];
    }
    
    /*44. Wildcard Matching
     * 野性匹配
     * *表示任意字符序列,可以为空。
     * ?表示任意一个字符*/
    public boolean isMatch3(String s, String p) {
        int m = s.length(),n = p.length();
        int i,j,k;
        boolean match[][] = new boolean[m+1][n+1];
        match[0][0] = true;
        for(j=1;j<=n;j++)
            if(p.charAt(j-1)!='*')
                break;
            else match[0][j] = true;
        for(i=1;i<=m;i++)
            for(j=1;j<=n;j++){
                if(s.charAt(i-1)==p.charAt(j-1) || p.charAt(j-1)=='?')
                    match[i][j] = match[i-1][j-1];
                else if(p.charAt(j-1)=='*'){
                    match[i][j] = match[i-1][j-1] || match[i-1][j] || match[i][j-1];
                }
            }
        return match[m][n];
    }
    
    public boolean isMatch4(String s, String p) {
    	int m = s.length(),n = p.length();
        int i,j;
        boolean seq[] = new boolean[n+1];
        seq[0] = true;
        for(j=1;j<=n;j++)
            if(p.charAt(j-1)!='*')
                break;
            else seq[j] = true;
        for(i=1;i<=m;i++){
            boolean pre = seq[0];//[i-1][j-1]
            seq[0] = false;//[i][j-1]
            for(j=1;j<=n;j++){
                boolean cur = seq[j];//[i-1][j]
                if(s.charAt(i-1)==p.charAt(j-1) || p.charAt(j-1)=='?')
                    seq[j] = pre;
                else if(p.charAt(j-1)=='*')
                    seq[j] = seq[j-1] || pre || cur;
                else seq[j] = false;
                pre = cur;
            }
        }
        return seq[n];
    }
    
    /*146. LRU Cache
     * 实现LRU Cache 的get和put方法
     * */
    LinkedHashMap<Integer,Integer> LRUCache;
    int capcity;
    public void LRUCache(int capcity) {
    	LRUCache = new LinkedHashMap<>();
    	this.capcity = capcity;
    }
    public int get(int key) {
    	if(!LRUCache.containsKey(key)) return -1;
    	int val = LRUCache.get(key);
    	LRUCache.remove(key);
    	LRUCache.put(key, val);
    	return val;
    }
    public void put(int key,int value) {
    	if(LRUCache.containsKey(key))
    		LRUCache.remove(key);
    	else if(LRUCache.size()==capcity)
    		LRUCache.remove(LRUCache.keySet().iterator().next());
    	LRUCache.put(key, value);
    }
    
    public int ringWalk(int n) {
    	if(n==0)
    		return 0;
    	int num[][] = new int[n][9];
    	int i,j;
    	num[0][1] = 1;
    	num[0][8] = 1;
    	for(i=0;i<n;i++) {
    		for(j=0;j<=9;j++) {
    			
    			
    		}
    	}
    	return 0;
    }
    
    public static void main(String[] args) {
		// TODO Auto-generated method stub
		LeetCode3 test = new LeetCode3();
		System.out.println(test.isMatch4("acdcb", "a*c?b"));
		
		
		System.out.println(test.isMatch1("aaa", ".*"));
		int AAA[] = {1,2,1,2,3};
		//System.out.println(test.subarraysWithKDistinct(AAA,2));
		//System.out.println(test.simplifyPath("//home/"));
		
		int sssss[][] = new int[5][5];
		for(int i=0;i<5;i++)
			System.out.println(sssss[i][3]);
		
		int aaaa[][] = {{2,3,1},{3,4,1}};
		Arrays.sort(aaaa[0]);
		for(int i=0;i<3;i++)
			System.out.println(aaaa[0][i]);
		for(int i=0;i<3;i++)
			System.out.println(aaaa[1][i]);
		
		System.out.println(test.reachingPoints(9, 10, 9, 19));
		
		String abcde = "abcdefegithidskdfajelg";
		String pateern = "abcde";
		System.out.println(abcde.indexOf(pateern,0));
		
		TreeNode ttt;
		TreeNode rrrr = test.new TreeNode(1);
		rrrr.right = test.new TreeNode(2);
		ttt = rrrr.right;
		ttt.right = test.new TreeNode(3);
		ttt = ttt.right;
		ttt.right = test.new TreeNode(4);
		ttt = ttt.right;
		ttt.right = test.new TreeNode(5);
		System.out.println(test.pathSum1(rrrr, 3));
		
		
		TreeNode xxxx = test.new TreeNode(1);
		xxxx.right = test.new TreeNode(15);
		System.out.println(test.sumNumbers(xxxx));
		
		TreeNode root = test.new TreeNode(7);
		root.left = test.new TreeNode(3);
		root.right = test.new TreeNode(15);
		TreeNode rr = root.right;
		rr.left = test.new TreeNode(9);
		rr.right = test.new TreeNode(20);
		
		BSTIterator teb = test.new BSTIterator(root);
		while(teb.hasNext())
			System.out.println(teb.next());
		
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
