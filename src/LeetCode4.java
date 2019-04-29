import java.util.*;

public class LeetCode4 {
	
	class TreeNode{
		TreeNode left=null,right=null;
		int val;
		public TreeNode(int val) {
			this.val = val;
		}
	}
	
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
    
    public static void main(String[] args) {
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
	
	/*public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner in = new Scanner(System.in);
		while (in.hasNextInt()) {// 注意while处理多个case
			int q = in.nextInt();
			for (int i = 0; i < q; i++) {
				int m = in.nextInt();
				int n = in.nextInt();
				if (m % 2 == 0 && n % 2 == 0) {
					System.out.println(-(n - m) / 2 + m);
				} else if (m % 2 == 1 && n % 2 == 1) {
					System.out.println((n - m) % 2 + m);
				} else if (m % 2 == 0 && n % 2 == 1) {
					System.out.println(-(n - m + 1) % 2);
				} else {
					System.out.println((n - m + 1) % 2);
				}
			}
		}

	}*/
}
