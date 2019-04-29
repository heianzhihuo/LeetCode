import java.util.*;
import java.util.Scanner;


public class LeetCode4 {
	
	class TreeNode{
		TreeNode left=null,right=null;
		int val;
		int count = 1;
		public TreeNode(int val) {
			this.val = val;
		}
	}
//    class TreeNode{
//        TreeNode left=null,right=null;
//        int val;
//        
//        public TreeNode(int val){
//            this.val = val;
//        }
//    }
	
	/*96. Unique Binary Search Trees
	 * 1-n的key构成的不同二叉搜索树的数目
	 * n个节点的不同结构的二叉树的数目*/
	public int numTrees(int n) {
        long s = 1;
        int i;
        for(i=1;i<=n;i++){
            s *= (n+i);
            s /= i;
        }
        return (int)(s/(n+1));
    }
	
	/*95. Unique Binary Search Trees II
	 * 生成所有1-n的节点组成的搜索二叉树*/
	public List<TreeNode> generateTrees(int n) {
        List<TreeNode> map[][] = new List[n][n];
        int d,i,j;
        TreeNode root;
        if(n==0)
            return new ArrayList<>();
        for(d=0;d<n;d++){
            for(i=0;i<n-d;i++){
                map[i][i+d] = new ArrayList<>();
                if(d==0){
                    root = new TreeNode(i+1);
                    root.left = null;root.right = null;
                    map[i][i].add(root);
                }else{
                    for(TreeNode right:map[i+1][i+d]){
                        root= new TreeNode(i+1);
                        root.left = null;
                        root.right = right;
                        map[i][i+d].add(root);
                    }
                    for(TreeNode left:map[i][i+d-1]){
                        root = new TreeNode(i+d+1);
                        root.left = left;
                        root.right = null;
                        map[i][i+d].add(root);
                    }
                    for(j=i+1;j<=i+d-1;j++)
                        for(TreeNode left:map[i][j-1])
                            for(TreeNode right:map[j+1][i+d]){
                                root = new TreeNode(j+1);
                                root.left = left;
                                root.right = right;
                                map[i][i+d].add(root);
                            }
                }
            }
        }
        return map[0][n-1];
    }
	
	public int maxCoins(int[] nums) {
        if(nums.length==0)
            return nums[0];
        int i;
        for(i=0;i<nums.length;i++)
        	shoot(nums,i,0);
        return max;
    }
    
    int max = 0;
    public void shoot(int[] nums,int x,int cur){
        if(nums.length==2){
            cur += nums[0]*nums[1] + Math.max(nums[0],nums[1]);
            if(cur>max)
                max = cur;
            return;
        }
        if(x==0)
            cur+=nums[0]*nums[1];
        else if(x==nums.length-1)
        	cur += nums[nums.length-1]*nums[nums.length-2];
        else {
        	cur += nums[x-1]*nums[x+1]*nums[x];
        }
        int remain[] = new int[nums.length-1];
        int i;
        for(i=0;i<x;i++)
        	remain[i] = nums[i];
        i++;
        for(;i<nums.length;i++)
        	remain[i-1] = nums[i];
        for(i=0;i<remain.length;i++)
        	shoot(Arrays.copyOf(remain, remain.length),i,cur);
    }
	
    /*330. Patching Array
     * 给定一个有序数组nums，
     * 最少需要向nums中添加几个数才能满足
     * 1-n之间的所有数都可以表示成nums中几个数的和
     * 思路：
     * 如何用最少的数组成1-n：1，2，4，8，16...规律
     * 假设在数组nums[0-i]之间的数之和为sum，且能组合出1-sum之间的数
     * 那么如果nums[i+1]=sum+1,0-i+1之间的数可以组合出sum+sum+1范围内的数
     * */
    public int minPatches(int[] nums, int n) {
        int i,count = 0;
        long sum = 0;
        for(i=0;i<nums.length;)
            if(sum>=n)
                break;
            else if(nums[i]<=sum+1){
                sum += nums[i];
                i++;
            }
            else{
                sum = sum+sum+1;
                count++;
            }
        while(sum<n){
            sum = sum+sum+1;
            count++;
        }
        return count;
    }
    
    /* 910. Smallest Range II
     * 给定一个数组A和一个数K，对数组A中每个数+K或者-K操作得到B
     * 最小的B中最大值和最小值的差值
     * 求最小的极差，
     * 对任意A[i]<A[j]，考虑肯定是A[i] up，而A[j] down
     * 对于由于数组A，A[0]+K,A[i]+K,A[i+1]-K,A[A.length-1]-K
     * I see
     * */
    public int smallestRangeII(int[] A, int K) {
        int n = A.length;
        if(n==1)
            return 0;
        Arrays.sort(A);
        int res = A[n-1]-A[0];
        for(int i=0;i<n-1;i++){
            int a = A[i],b = A[i+1];
            int high = Math.max(A[n-1]-K,a+K);
            int low = Math.min(A[0]+K,b-K);
            res = Math.min(res,high-low);
        }
        return res;
    }
    
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
    
    public static void main(String[] args) {
        LeetCode4 test = new LeetCode4();
    	int A[] = {2147483647,2147483647,2147483647,2147483647,2147483647,2147483647};
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
