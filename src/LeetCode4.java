import java.util.*;
import java.util.Scanner;


public class LeetCode4 {
	
	/*493. Reverse Pairs
	 * i<j && nums[i]>2*nums[j]
	 * */
	public int reversePairs(int[] nums) {
        return reversePairs(nums,0,nums.length-1);
    }
    
    public int reversePairs(int[] nums,int low,int high){
        if(low<high){
            int mid = (low+high)/2;
            int sum = reversePairs(nums,low,mid)+reversePairs(nums,mid+1,high);
            int i = low,j = mid+1;
            for(;i<=mid;i++){
                while(j<=high && nums[i]>nums[j]*2L)
                    j++;
                sum += (j-mid-1);
            }
            merge(nums,low,mid,high);
            return sum;
        }
        return 0; 
    }
    public void merge(int[] nums,int low,int mid,int high){
        int tmp[] = new int[high-low+1];
        int i = low,j = mid+1,k = 0;
        while(i<=mid || j<=high){
            if(i==mid+1 || (j<high+1 && nums[i]>nums[j])){
                tmp[k] = nums[j];
                j++;
            }else{
                tmp[k] = nums[i];
                i++;
            }
            k++;
        }
        for(k=0;k<high-low+1;k++)
            nums[low+k] = tmp[k];
    }
    
    /* 315. Count of Smaller Numbers After Self
     * counts[i] is the number of smaller elements to the right of nums[i]
     * 计算数组nums中在每个元素右侧且比它大的元素个数*/
    public List<Integer> countSmaller(int[] nums) {
        List<Integer> result = new ArrayList<>();
        int n = nums.length;
        if(n==0)
            return result;
        TreeNode root = new TreeNode(nums[n-1]);
        result.add(0);
        int i;
        for(i=n-2;i>=0;i--)
            result.add(insert(root,nums[i]));
        Collections.reverse(result);
        return result;
    }
    public int insert(TreeNode root,int val){
        int count = 0;
        while(true){
            if(val<=root.val){
                root.count++;
                if(root.left==null){
                    root.left = new TreeNode(val);
                    break;
                }
                else
                    root = root.left;
            }else{
                count+=root.count;
                if(root.right==null){
                    root.right = new TreeNode(val);
                    break;
                }else
                    root = root.right;
            }
        }
        return count;
    }
    class TreeNode{
        TreeNode left=null,right=null;
        int val;
        int count = 1;
        public TreeNode(int val){
            this.val = val;
        }
    }
    
    /*922. Sort Array By Parity II
     * 将数组重新排序，使得下标位置为奇数，偶数下标位置为偶数*/
    public int[] sortArrayByParityII(int[] A) {
        int i,j,n = A.length;
        j = 1;
        for(i=0;i<n;i+=2)
            if(A[i]%2==1){
                for(;j<n;j+=2)
                    if(A[j]%2==0)
                        break;
                int t = A[i];
                A[i] = A[j];
                A[j] = t;
            }
        return A;
    }
    
    /* 999. Available Captures for Rook
     * 国际象棋棋盘上有rook,empty,bishops,pawns，分别用'R','.','B','p'表示
     * rook只能走直线，在一步之内rook能吃到几个pawns
     * 只有一个rook
     * */
    public int numRookCaptures(char[][] board) {
        int x=-1,y=-1,i,j;
        int m = board.length,n = board[0].length;
        for(i=0;i<m;i++)
            for(j=0;j<n;j++)
                if(board[i][j]=='R'){
                    x = i;
                    y = j;
                    break;
                }
        if(x==-1 || y==-1)
            return 0;
        int count = 0;
        for(i=x;i<n;i++)
            if(board[i][y]=='B')
                break;
            else if(board[i][y]=='p'){
                count++;
                break;
            }
        for(i=x;i>=0;i--)
            if(board[i][y]=='B')
                break;
            else if(board[i][y]=='p'){
                count++;
                break;
            }
        for(j=y;j>=0;j--)
            if(board[x][j]=='B')
                break;
            else if(board[x][j]=='p'){
                count++;
                break;
            }
        for(j=y;j<m;j++)
             if(board[x][j]=='B')
                break;
            else if(board[x][j]=='p'){
                count++;
                break;
            }
        return count;
    }
	
    /* 985. Sum of Even Numbers After Queries
     * 给定数组A，和给定的queries操作序列
     * queries中第i个操作的val=queries[i][0],index=queries[i][1]
     * 把val加到A的第index个数中
     * 返回每次queries后A数组中偶数之和
     * */
    public int[] sumEvenAfterQueries(int[] A, int[][] queries) {
        int[] result = new int[queries.length];
        int sum = 0;
        int i;
        for(i=0;i<A.length;i++)
            if(A[i]%2==0)
                sum += A[i];
        for(i=0;i<queries.length;i++){
            int val = queries[i][0];
            int index = queries[i][1];
            if(A[index]%2==0)
                sum -= A[index];
            A[index] += val;
            if(A[index]%2==0)
                sum += A[index];
            result[i] = sum;
        }
        return result;
    }
    
    /* 1002. Find Common Characters
     * 找到在所有字符串中都出现的字符*/
    public List<String> commonChars(String[] A) {
        int count[][] = new int[26][A.length];
        for(int i=0;i<A.length;i++){
            for(int j=0;j<A[i].length();j++)
                count[A[i].charAt(j)-'a'][i]++;
        }
        List<String> result = new ArrayList<>();
        for(int i=0;i<26;i++){
            int num = 0;
            for(int j=0;j<A.length;j++){
                if(count[i][j]==0){
                    num=0;
                    break;
                }
                if(num==0 || count[i][j]<num)
                    num = count[i][j];
            }
            for(int j=0;j<num;j++)
                result.add((char)(i+'a')+"");
        }
        return result;
    }
    /*670. Maximum Swap
     * 给一个整数，至多交换一次，交换整数的两位，使得交换后最大*/
    public int maximumSwap(int num) {
        ArrayList<Integer> res = new ArrayList<>();
        ArrayList<Integer> tmp = new ArrayList<>();
        int i,j,t = num;
        while(t>0){
            res.add(t%10);
            tmp.add(t%10);
            t /= 10;
        }
        Collections.sort(res);
        for(i=res.size()-1;i>=0;i--)
            if(res.get(i)>tmp.get(i))
                break;
        if(i==-1)
            return num;
        for(j=0;j<i;j++)
            if(res.get(i)==tmp.get(j))
                break;
        t = tmp.get(i);
        tmp.set(i,tmp.get(j));
        tmp.set(j,t);
        t = 0;
        for(i=tmp.size()-1;i>=0;i--){
            t = t*10 + tmp.get(i);
        }
        return t;
    }
    
    /* 18. 4Sum
     * 经典的2Sum，3Sum，4Sum问题
     * 给定数组中4个数之和等于目标值的选择数*/
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<>();
        int n = nums.length;
        if(n<4)
            return res;
        Arrays.sort(nums);
        int i,j,p,q,x,t;
        i = 0;
        while(i<n-3){
            j = i+1;
            while(j<n-2){
                p = j+1;q = n-1;
                t = target-nums[i]-nums[j];
                while(p<q){
                    if(nums[p]+nums[q]>t)
                        q--;
                    else if(nums[p]+nums[q]<t)
                        p++;
                    else{
                        ArrayList<Integer> tmp = new ArrayList<Integer>();
                        tmp.add(nums[i]);
                        tmp.add(nums[j]);
                        tmp.add(nums[p]);
                        tmp.add(nums[q]);
                        res.add(tmp);
                        x = nums[p];
                        while(p<q && nums[p]==x)p++;
                        q--;
                    }
                }
                x = nums[j];
                while(j<n-2 && nums[j]==x)j++;
            }
            x = nums[i];
            while(i<n-3 && nums[i]==x)i++;
        }
        return res;
    }
    
    /*942. DI String Match
     * 给定一个数组S，生成一个1-N的排列A
     * 使得 ifS[i]=='I',then A[i]<A[i+1]
     * ifS[i]=='D' then  A[i]>A[i+1] 
     * 思路很简单，就是如果S[0]=='I' 则A[0] = 0
     * 否则，A[0]= N;
     * */
    public int[] diStringMatch(String S) {
        int i,j,k;
        i = 0;
        j = S.length();
        int result[] =new int[j+1];
        for(k=0;k<S.length();k++)
            if(S.charAt(k)=='I'){
                result[k] = i;
                i++;
            }else{
                result[k] = j;
                j--;
            }
        result[k] = i;
        return result;
    }
    
    /*279. Perfect Squares
     * 讲n拆分成最少数目的完全平方数的和
     * 动态规划*/
    public int numSquares(int n) {
        int result[] = new int[n+1];
        int i,j;
        result[0] = 0;
        for(i=1;i<=n;i++){
            int min = i;
            for(j=2;j*j<=i;j++)
                if(result[i-j*j]+1<min)
                    min = Math.min(result[i-j*j]+1,min);
            result[i] = min;
        }
        return result[n];
    }
    /*310. Minimum Height Trees
     * 给定一个无环无向图，选择一个顶点作为根节点，使得树的高度最小
     * 思路：找到最长的路径，以这个路径的中点作为根
     * 最长路径的求解：先找到离一个节点最远的节点，然后求以最远节点开始的路径
     * 
     * 这题就是找最中间的两个点，还有一种思路是一直把叶节点删除，
     * 直到图中剩1个或者2个点为止
     * */
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        adj = new List[n];
        int i;
        for(i=0;i<n;i++)
            adj[i] = new ArrayList<>();
        for(i=0;i<edges.length;i++){
            adj[edges[i][0]].add(edges[i][1]);
            adj[edges[i][1]].add(edges[i][0]);
        }
        visited = new boolean[n];
        List<Integer> path = longestPath(0);
        visited = new boolean[n];
        List<Integer> longepath = longestPath(path.get(0));
        List<Integer> result = new ArrayList<>();
        i = longepath.size();
        if(i%2==0){
            result.add(longepath.get(i/2));
            result.add(longepath.get(i/2-1));
        }else
            result.add(longepath.get(i/2));
        return result;
        
    }
    List<Integer>[] adj;
    boolean []visited;
    List<Integer> longestPath(int v){
        List<Integer> result=null;
        visited[v] = true;
        for(int c:adj[v])
            if(!visited[c]){
                List<Integer> tmp = longestPath(c);
                if(result==null || result.size()<tmp.size())
                    result = tmp;
            }
        if(result==null)
            result = new ArrayList<>();
        result.add(v);
        return result;        
    }
    
    /*207. Course Schedule
     * 课程安排问题：拓扑排序问题
     * 给定标签0-n-1的n门课程，其中[0,1]表示，1是0的先修课*/
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        int n = numCourses;
        adj = new List[n];
        int i;
        for(i=0;i<n;i++)
            adj[i] = new ArrayList<>();
        for(i=0;i<prerequisites.length;i++)
            adj[prerequisites[i][1]].add(prerequisites[i][0]);
        visited = new boolean[n];//用于判断课程是否已经完成
        finished = new boolean[n];//
        for(i=0;i<n;i++){
            if(DFS(i))
                return false;
        }
        return true;
    }
    boolean []finished;
    boolean DFS(int v){
        if(finished[v])
            return false;
        else if(visited[v])
            return true;
        visited[v] = true;
        for(int c:adj[v])
            if(DFS(c))
                return true;
        finished[v] = true;
        return false;
    }
    
    public static void main(String[] args) {
        LeetCode4 test = new LeetCode4();
    	int A[] = {2147483647,2147483647,2147483647,2147483647,2147483647,2147483647};
    	int prerequisites[][] = {{0,1},{0,2},{1,2}};
    	System.out.println(test.canFinish(3, prerequisites));
    	
    	
    	
    	System.out.println(test.reversePairs(A));
    	for(int x:A)
    		System.out.println(x);
    	int x = 1;
    	switch(x) {
    	case 1:
    		int n = 0;
    		break;
    	}
    	
    	
    	Scanner in = new Scanner(System.in);
        while (in.hasNextInt()) {// 注意，如果输入是多个测试用例，请通过while循环处理多个测试用例
        	
            int d = in.nextInt();
            String a = in.nextLine();
            String b = in.nextLine();
            String res = "";
            String aa[] = a.split(",");
            String bb[] = b.split(",");
            int n1 = aa.length;
            int n2 = bb.length;
            int i,j,k;
            i = 0;
            while(i+d<n1 && i+d<n2){
                for(k=0;k<d;k++)
                    res += aa[i+k] + ",";
                for(k=0;k<d;k++)
                    res += bb[i+k] + ",";
                i += d;
            }
            if(i+d<n1){
                for(k=0;k<d;k++)
                    res += aa[i+k] + ",";
                for(k=0;k<n2-d;k++)
                    res += bb[i+k] + ",";
                i+=d;
                for(;i<n1;i++)
                    res += aa[i+k] + ",";
            }else{
                for(k=0;k<n1-d;k++)
                    res += aa[i+k] + ",";
                for(;i<n2;i++)
                    res += bb[i+k] + ",";
            }
            System.out.println(res.substring(res.length()-1));
        }
    }
	
}
