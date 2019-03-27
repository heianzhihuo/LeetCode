import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Stack;


/**
 * @author WenWei
 * @date 2018年12月3日
 * @time 下午10:18:40
 */
public class LeetCode {
	
	/* 503
	 * 循环数组中下一个比这个数大的数
	 * Input: [1,2,1]
	 * Output: [2,-1,2]
	 * print the Next Greater Number for every element. 
	 * number to its traversing-order next in the array, which means 
	 * you could search circularly to find its next greater number. 
	 * If it doesn't exist, output -1 for this number*/
	static int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] result = new int[n];
        int i,j,k;
        k = 0;//最大数下标
        for(i=0;i<n;i++) {
        	result[i] = -1;
        	if(nums[i]>nums[k])
        		k = i;
        }
        if(n<=1)
        	return result;
        for(i=k==0?n-1:k-1;i!=k;i=i==0?n-1:i-1)
        {
            j = i==n-1?0:i+1;
            while(j!=-1 && nums[j]<=nums[i])
            	j = result[j];
            result[i] =  j;
        }
        for(i=0;i<n;i++)
        	result[i] = result[i]!=-1?nums[result[i]]:-1; 
    	return result;
    }
	
	/* LeetCode 12
	 * 数字转罗马数字
	 * 
	 * I             1
	 * V             5
	 * X             10
	 * L             50
	 * C             100
	 * D             500
	 * M 			 1000*/
	public static String intToRoman(int num) {
		String[] table = {"I","V","X","L","C","D","M"};
        String result = "";
        String tmp;
        int i,j,k;
        k = 0;
        while(num>0) {
        	i = num%10;
        	num = num/10;
        	if(1<=i && i<=3) {
        		for(j=0;j<i;j++)
        			result = table[k*2] + result;
        	}else if(i==4) {
        		result = table[k*2]+table[k*2+1] + result;
        	}else if(i==5) {
        		result = table[k*2+1] + result;
        	}else if(6<=i && i<=8) {
        		for(j=0;j<i-5;j++)
        			result = table[k*2] + result;
    			result = table[k*2+1] + result;
        	}else if(i==9) {
        		result = table[k*2]+table[k*2+2] + result;
        	}
        	k++;
        }
        return result;
    }
	/* LeetCode 455 
	 * 分配饼干
	 * g[] 第i个小孩需要的最小饼干，s[]为各个饼干的大小
	 * 目的是给最多的小孩分配饼干
	 * */
	public static int findContentChildren(int[] g, int[] s) {
        int result = 0;
        if(g.length==0||s.length==0)
			return result;
        int i,j;
        Arrays.sort(s);
        Arrays.sort(g);
        i = 0;
        j = 0;
        while(i<g.length && j<s.length) {
        	if(g[i]<=s[j]) {
        		result++;
        		i++;
        		j++;
        	}else {
        		j++;
        	}
        }
        
        return result;
    }
	
	/* 78 子集问题
	 * 求一个集合的所有子集
	 * */
	public static List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        int k,i,j,n = nums.length;
        List<Integer> tmp = new ArrayList<>();
        result.add(tmp);
        for(k=0;k<n;k++) {
        	//形成k大的子集
        	int[] index = new int[k+2];
        	j = 0;
        	index[0] = -1;
        	while(true) {
        		index[j]++; 
        		
        		if(k-j+index[j]>=n) {
        			//需要回溯
        			//当前位分配的下标超过上限
        			index[j] = -1;
        			j--;//返回上一个位置
        		}else {
        			index[j+1] = index[j];
        			j++;
        		}
        		if(j>k) {
        			//已填满，则保存
        			tmp = new ArrayList<>();
        			for(i=0;i<k+1;i++)
        				tmp.add(nums[index[i]]);
        			result.add(tmp);
        			if(index[0] == n-k-1) {
        				break;//是最后一个
        			}
        			j--;
        		}
        	}
        }
        return result;
    }
	
	/* 733.填充问题
	 * 从某个位置开始，
	 * 将所有和这个位置值相同的位置填充为新的颜色值
	 * */
	public static int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        int m = image.length,n = image[0].length;
		int[][] result = new int[m][n];
		int i,j,k,x,y;
		List<Integer> current_x = new ArrayList<>();
		List<Integer> current_y = new ArrayList<>();
		List<Integer> next_x = new ArrayList<>();//队列
		List<Integer> next_y = new ArrayList<>();
		current_x.add(sr);
		current_y.add(sc);
		k = image[sr][sc];
		for(i=0;i<m;i++)
			for(j=0;j<n;j++)
				result[i][j] = image[i][j]; 
		if(k==newColor)
			return result;
		while(!current_x.isEmpty()) {
			next_x = new ArrayList<>();
			next_y = new ArrayList<>();
			for(i=0;i<current_x.size();i++) {
				//遍历刚才遍历过的点
				x = current_x.get(i);
				y = current_y.get(i);
				result[x][y] = newColor;
				if(y>0 && result[x][y-1]==k) {
					next_x.add(x);
					next_y.add(y-1);
				}
				if(y<n-1 && result[x][y+1]==k) {
					next_x.add(x);
					next_y.add(y+1);
				}
				if(x>0 && result[x-1][y]==k) {
					next_x.add(x-1);
					next_y.add(y);
				}
				if(x<m-1 && result[x+1][y]==k) {
					next_x.add(x+1);
					next_y.add(y);
				}
			}
			current_x = next_x;
			current_y = next_y;
		}
		
		return result;
    }
	
	/* 679.数组的度
	 * 数组的度定义为出现频率最大的元素
	 * 任务是找到最小的连续子数组，其度和原数组相同
	 * 返回这样的数组的长度
	 * */
	public static int findShortestSubArray(int[] nums) {
        int result = nums.length;
        HashMap<Integer,Integer> counter = new HashMap<>();
        int i,j,k,maxf;
        //对数组各个数计数
        for(int t:nums) {
        	if(counter.containsKey(t))
        		counter.replace(t, counter.get(t)+1);
        	else
        		counter.put(t, 1);
        }
        //对各个数按出现频率从大到小排列
        List<Integer> index = new ArrayList<>(counter.keySet());
        index.sort(new Comparator<Integer>() {
			@Override
			public int compare(Integer o1, Integer o2) {
				// TODO Auto-generated method stub
				return counter.get(o2)-counter.get(o1);
			}
		});
        k = index.get(0);
        maxf = counter.get(index.get(0));
        for(int t:index) {
        	if(counter.get(t)==maxf) {
        		for(i=0;i<nums.length;i++)
        			if(nums[i]==t)
        				break;
        		for(j=nums.length-1;j>=0;j--)
        			if(nums[j]==t)
        				break;
        		if(j+1-i<result)
        			result = j+1-i;
        		if(result==maxf)
        			break;
        	}else
        		break;
        }
        
        return result;
    }
	
	/* 765.夫妻携手问题
	 * N对夫妻围绕2N个座位坐，想要手牵手
	 * 想要直到最小的交换数，使得所有夫妻都相邻
	 * 一个交换在任意两个之间
	 * row[i]表示第i个人的当前所在的位置
	 * 
	 * ps.
	 * 想法：生成有向图，找出不相交的环，
	 * 每个环至少要n/2-1次交换，
	 * n为环中顶点数
	 * */
	public int minSwapsCouples(int[] row) {
		int result = 0;
		int i;
		if(row.length==0)
			return 0;
		for(i=0;i<row.length-1;i++) {
			if((row[i]%2==0 && row[i+1]==row[i]+1) 
					|| (row[i]%2==1 && row[i+1]==row[i]-1))
				result++;
		}
		if((row[0]%2==0 && row[row.length-1]==row[0]+1)||
				(row[0]%2==1 && row[row.length-1]==row[0]-1))
			result++;
        return row.length/2-result;
    }
	
	/* 598.范围加法
	 * 给定m*n的矩阵M初始化为0
	 * 2D的数组operations，有很多2维的元素
	 * operation的元素用[a,b]表示
	 * 每一个元素表示将矩阵0<=i<a && 0<=j<b 区间的值加1
	 * 有若干的操作
	 * 最后返回操作结束后矩阵中最大值的数目
	 * Input: 
		m = 3, n = 3
		operations = [[2,2],[3,3]]
		Output: 4
		Explanation: 
		Initially, M = 
		[[0, 0, 0],
		 [0, 0, 0],
		 [0, 0, 0]]
		
		After performing [2,2], M = 
		[[1, 1, 0],
		 [1, 1, 0],
		 [0, 0, 0]]
		
		After performing [3,3], M = 
		[[2, 2, 1],
		 [2, 2, 1],
		 [1, 1, 1]]
	 * */
	public static int maxCount(int m, int n, int[][] ops) {
		int i;
        for(i=0;i<ops.length;i++) {
        	if(ops[i][0]<m)
        		m = ops[i][0];
        	if(ops[i][1]<n)
        		n = ops[i][1];
        }
        return m*n;
    }
	
	/* 387.第一个不重复的字母
	 * */
	
	public static int firstUniqChar(String s) {
        int count[] = new int[26];
        int i;
        for(i=0;i<s.length();i++){
        	int k = s.charAt(i) - 'a';
            if(count[k]==1){
                count[k] = -1;
            }else if(count[k]==0)
                count[k] = 1;
        }
        for(i=0;i<s.length();i++) {
        	if(count[s.charAt(i)-'a']>0)
        		return i;
        }
        return -1;
    }
	
	/* 241.不同计算方式
	 * 给一个字符串算术表达式添加括号
	 * 计算出不同的值
	 * 返回所有不同的值
	 * 表达式中只有+、-、*
	 * */
	public static List<Integer> diffWaysToCompute(String input) {
        List<Integer> result = new ArrayList<>();
        int i,j,k,m,n = input.length();
        //int matrix[][] = new int[n][n];
        //HashSet<Integer> matrix[][] = new HashSet[n][n];
        int nums[] = new int[n];
        int op[] = new int[n];
        m = 0;//运算符个数
        j = 0;//上一个数字所在的下标
        for(i=0;i<n;i++) {
        	if(input.charAt(i)<'0') {
        		switch(input.charAt(i)) {
            	case '+':
            		op[m] = 0;
            		break;
            	case '-':
            		op[m] = 1; 
            		break;
            	case '*':
            		op[m] = 2;
            		break;
            	}
            	//遇到操作符
        		//将操作符前的操作数保存起来
        		nums[m] = Integer.valueOf(input.substring(j,i));
        		j = i+1;
        		m++;
        	}
        }
        nums[m] =  Integer.valueOf(input.substring(j,i));
        List<Integer> matrix[][] = new List[m+1][m+1];
        for(i=0;i<=m;i++) {
        	for(j=i;j<=m;j++)
        		matrix[i][j] = new ArrayList<>();
        	matrix[i][i].add(nums[i]);
        }
        for(k=1;k<=m;k++) {
        	for(i=0;i<m+1-k;i++) {
        		for(j=i;j<i+k;j++) {
        			for(int a:matrix[i][j])
        				for(int b:matrix[j+1][i+k]) {
        					int c = 0;
        					switch(op[j]) {
        					case 0:
        						c = a+b;
        						break;
        					case 1:
        						c = a-b;
        						break;
        					case 2:
        						c = a*b;
        						break;
        					}
        					matrix[i][i+k].add(c);
        				}
        		}
        	}
        }
        //每次计算
        result.addAll(matrix[0][m]);
        return result;
    }
	
	/* 189.数组循环移位
	 * 向右循环移动k位
	 * */
	public static void rotate(int[] nums, int k) {
        int i,j,tmp;
        int n = nums.length;
        int a,b;
        if(n>k) {
        	a = n;
        	b = k;
        }else {
        	a = k;
        	b = n;
        }
        //辗转相除法求最大公约数a
        while(b>0) {
        	tmp = b;
        	b = a%b;
        	a = tmp;
        }
        for(i=0;i<a;i++) {
        	j = i;
        	b = j+n+(-k)%n;
        	tmp = nums[b];
        	
        	while(j>i) {
        		b = (j+k)%n;
        		nums[b] = nums[j]; 
        		j = b;
        	}
        	nums[j] = tmp;   
        	
        }
	}
	
	/* 756.金字塔转移矩阵
	 * 
	 * */
	public static boolean pyramidTransition(String bottom, List<String> allowed) {
        boolean result = false;
        
        
        return result;
    }

	/* 144.二叉树前序遍历
	 * 返回二叉树前序遍历序列
	 * */
	public class TreeNode {
		      int val;
		      TreeNode left;
		      TreeNode right;
		      TreeNode(int x) { val = x; }
	 }
	List<Integer> preorder;
	public void porderTraversal(TreeNode root) {
		if(root!=null) {
			preorder.add(root.val);
			porderTraversal(root.left);
			porderTraversal(root.right);
		}
	}
	
	public List<Integer> preorderTraversal(TreeNode root) {
        preorder = new ArrayList<>();
        porderTraversal(root);
        return preorder;
    }
	
	/* 530.二叉搜索树的最小差值
	 * 二叉搜索树中任意两个节点之间的最小值
	 * */
	public int getMinimumDifference(TreeNode root) {
        int min = Integer.MAX_VALUE;
        int a,b;
        TreeNode p;
        if(root==null)
        	return min;
        if(root.left!=null) {
        	a = getMinimumDifference(root.left);
        	p = root.left;
        	while(p.right!=null)
        		p = p.right;
        	b = root.val-p.val;
        	if(a<min)
        		min = a;
        	if(b<min)
        		min = b;
        }
        if(root.right!=null) {
        	a = getMinimumDifference(root.right);
        	p = root.right;
        	while(p.left!=null)
        		p = p.left;
        	b = p.val-root.val;
        	if(a<min)
        		min = a;
        	if(b<min)
        		min = b;
        }
        return min;
    }
	
	/* 5.最长回文子串
	 * 返回字符串中的最长回文子串
	 * */
	public int expandAroundCenter(String s,int left,int right) {
		int L = left,R = right;
		while(L>=0 && R<s.length() && s.charAt(L)==s.charAt(R)) {
			L--;
			R++;
		}
		return R-L-1;
	}
	
	public String longestPalindrome(String s) {
        if(s==null || s.length()<1)
        	return "";
        int start = 0,end = 0;
        for(int i=0;i<s.length();i++) {
        	int len1 = expandAroundCenter(s,i,i);
        	int len2 = expandAroundCenter(s, i, i+1);
        	int len = len1;
        	if(len2>len1)
        		len = len2;
        	if(len>end-start) {
        		start = i-(len-1)/2;
        		end = i+(len)/2;
        	}
        }
        return s.substring(start,end+1);
    }
	
	/* 67.二进制相加
	 * 二进制字符串相加*/
	public String addBinary(String a, String b) {
        String result = "";
        int m = a.length()-1,n = b.length()-1;
        char flag = 0;
        char tmp;
        while(m>=0 || n>=0) {
        	if(m>=0 && n>=0) {
        		tmp = (char)(flag^(a.charAt(m)-'0')^(b.charAt(n)-'0') +'0');
        		if(flag+a.charAt(m)-'0'+b.charAt(n)-'0'>1)
        			flag = 1;
        		else
        			flag = 0;
        		m--;
        		n--;
        	} else if(m>=0) {
        		tmp = (char)(flag^(a.charAt(m)-'0')+'0');
        		if(flag+a.charAt(m)-'0'>1)
        			flag = 1;
        		else
        			flag = 0;
        		m--;
        	} else {
        		tmp = (char)(flag^(b.charAt(n)-'0')+'0');
        		if(flag+b.charAt(n)-'0'>1)
        			flag = 1;
        		else
        			flag = 0;
        		n--;
        	}
        	result = tmp+result;
        }
        if(flag>0)
        	result = (char)(flag+'0') + result;
        return result;
    }
	
	/* 165.版本号比较
	 * 如果version1>version2, return 1
	 * if version1<version2, return -1
	 * other return 0;
	 * 版本号实例
	 * 0.1,1.1,1.0.1,1,7.5.2.4,7.5.3
	 * 
	 * */
	public int compareVersion(String version1, String version2) {
        int i=0,j=0;
        //int flag1=1,flag2=1;
        boolean flag1 = true,flag2 = true;
        int m = version1.length(),n = version2.length();
        while(i<m && j<n) {
        	if(flag1 && version1.charAt(i)=='0')
        		i++;
        }
        if(m==n)
        	return 0;
        else if(m>n)
        	return 1;
        else return -1;
    }
	
	/* 538.将搜索二叉树转换成更大的树
	 * 每个节点加上比这个节点大的所有节点的值*/
	int prev_node_val = 0;
	public TreeNode convertBST(TreeNode root) {
        if(root==null)
        	return root;
        root.right = convertBST(root.right);
        root.val += prev_node_val;
        prev_node_val = root.val;
        root.left = convertBST(root.left);
        return root;
    }
	
	/* 204.计算素数个数
	 * 小于n的素数个数
	 * */
	public int countPrimes(int n) {
        if(n<=2)
        	return 0;
		boolean primes[] = new boolean[n];
        int i,j,k=0;
        primes[0] = true;
        primes[1] = true;
        for(i=2;i*i<n;i++)
        	if(!primes[i]) 
        		for(j=2;j*i<n;j++) 
	        		primes[i*j] = true;
        for(i=2;i<n;i++)
        	if(!primes[i])
        		k++;
        return k;
    }
	
	/* 714.股票购买事务
	 * prices[i],第i天的股票价格
	 * fee表示每次事务(买入一次，再卖出一次)所需的代价
	 * 每次只能有一支股票,买入前必需卖出当前的股票
	 * */
	public int maxProfit(int[] prices, int fee) {
        int n = prices.length;
        int i,j,k;
        int profit = 0;
        int in1 = Integer.MAX_VALUE,out1 = -1,in2 = Integer.MAX_VALUE,out2 = -1;
        int flag0 = 0,flag1 = 0;
        for(i=0;i<n;i++) {
        	if(flag1==0) {
        		if(prices[i]<in2) {
        			//下降
        			in2 = prices[i];
        		}else if(prices[i]>out2) {
        			flag1 = 1;
        			//开始上升
        			out2 = prices[i];
        		}
        	}else if(flag1==1) {
        		if(prices[i]>out2) {
        			
        		}
        	}
        }
		
		return profit;
    }
	
	/* 954.偶数对数组
	 * A[2 * i + 1] = 2 * A[2 * i] 对任意0<=i<len(A)/2成立
	 * 可以随意调换数组顺序
	 * */
	public boolean canReorderDoubled(int[] A) {
        int i;
        int flag = 1;
        if(A==null || A.length==0)
        	return true;
        if(A.length%2==1)
        	return false;
        int freq[] = new int[100000];
        for(i=0;i<A.length;i++) {
        	if(A[i]<0) {
        		flag = -flag;
        		A[i] = -A[i]; 
        	}
        	freq[A[i]]++;
        }
        if(flag<0)
        	return false;
        for(i=0;i<100000;i++) {
        	if(freq[i]>0) {
        		freq[i*2] -= freq[i];
        		freq[i] = 0;
        		if(freq[i*2]<0)
        			return false;
        	}
        }
		return true;
    }
	
	public int lengthLongestPath(String input) {
        int max_length = 0;
        //int current_level = -1;
        Stack<Integer> current = new Stack<>();
        int current_length = 0;
        boolean flag = false;
        String tmp;
        int i,j,k,t;
        i = 0;
        while(i<input.length()) {
        	//找到一个串，即一个文件夹
        	j = input.indexOf('\n',i);
        	if(j==-1)
        		j = input.length();
        	tmp = input.substring(i,j);
        	k = tmp.lastIndexOf('\t')+1;//表示所在的层次
        	//System.out.println(current.size());
        	if(k<current.size()) {
        		//当前层次比栈顶层次低
        		//保存当前路径长度
        		if(current_length>max_length)
        			max_length = current_length;
        		while(k<current.size()) 
        			//弹栈，直到当前层次和栈顶层次相同
        			current_length -= current.pop();
        	}
        	t = j-i-k+1;
        	current.add(t);
    		current_length += t;
        	i = j+1;
        }
        if(current_length>max_length)
        	max_length = current_length;
		return max_length-1;
    }
	
	/* 494.目标和
	 * 在给定的序列添加+和-号，是得表达式结果等于S的方法数
	 * */
	public int findTargetSumWays(int[] nums, int S) {
        HashMap<Integer,Integer> current = new HashMap<>();
        HashMap<Integer,Integer> next;
        current.put(0, 1);
        int i;
        for(i=0;i<nums.length;i++) {
        	next = new HashMap<>();
        	for(Entry<Integer,Integer> entry:current.entrySet()) {
            	int x = entry.getKey();
            	int y = entry.getValue();
            	int a = x+nums[i];
            	if(next.containsKey(a)) {
            		next.put(a,next.get(a)+y);
            	}else {
            		next.put(a, y);
            	}
            	a = x-nums[i];
            	if(next.containsKey(a)) {
            		next.put(a,next.get(a)+y);
            	}else {
            		next.put(a, y);
            	}
            }
        	current = next;
        }
        if(current.containsKey(S))
        	return current.get(S);
		return 0;
    }
	
	/* 916.字子集
	 * 
	 * */
	public List<String> wordSubsets1(String[] A, String[] B) {
		List<String> result = new ArrayList<>();
		int Amap[][] = new int[A.length][26];
		int Bmax[] = new int[26];
		int Bcurrent[];
		int i,j;
		for(i=0;i<A.length;i++) 
			for(char c:A[i].toCharArray())
				Amap[i][c-'a'] += 1;
		for(i=0;i<B.length;i++) {
			Bcurrent = new int[26];
			for(char c:B[i].toCharArray())
				Bcurrent[c-'a'] += 1;
			for(j=0;j<26;j++)
				if(Bmax[j]<Bcurrent[j])
					Bmax[j] = Bcurrent[j]; 
		}
		for(i=0;i<A.length;i++) {
			boolean flag = true;
			for(j=0;j<26;j++)
				if(Amap[i][j]<Bmax[j]) {
					flag = false;
					break;
				}
			if(flag)
				result.add(A[i]);
		}
		return result;
	}
	public List<String> wordSubsets(String[] A, String[] B) {
		List<String> result = new ArrayList<>();
        HashMap<Character,Integer> Amap[] = new HashMap[A.length];
        HashMap<Character,Integer> Bmax = new HashMap<>();
        HashMap<Character,Integer> current;
        int i;
        char tmp[];
        for(i=0;i<A.length;i++) {
        	tmp = A[i].toCharArray();
        	Amap[i] = new HashMap<>(); 
        	for(char c:tmp) {
        		if(Amap[i].containsKey(c)) {
        			Amap[i].put(c, Amap[i].get(c)+1);
        		}else{
        			Amap[i].put(c, 1);
        		}
        	}
        }
        for(i=0;i<B.length;i++) {
        	tmp = B[i].toCharArray();
        	current = new HashMap<>();
        	for(char c:tmp) {
        		if(current.containsKey(c)) {
        			current.put(c, current.get(c)+1);
        		}else {
        			current.put(c, 1);
        		}
        	}
        	for(Entry<Character,Integer> entry:current.entrySet()) {
        		char x = entry.getKey();
    			int n = entry.getValue();
    			if(!Bmax.containsKey(x) || Bmax.get(x)<n)
    				Bmax.put(x, n);
        	}
        }
        for(i=0;i<A.length;i++) {
        	boolean flag = true;
    		for(Entry<Character,Integer> entry:Bmax.entrySet()) {
    			char x = entry.getKey();
    			int n = entry.getValue();
    			if(!Amap[i].containsKey(x) || Amap[i].get(x)<n) {
    				flag = false;
    				break;
    			}
        	}
        	if(flag)
        		result.add(A[i]);
        }
        return result;
    }
	
	
	/* 273.整数转换成英文单词
	 *  将整数转换成英文读法
	 * */
	public String numberToWords(int num) {
        String result = "";
        String tens[] = {"Twenty","Thirty","Forty","Fifty",
        		"Sixty","Seventy","Eighty","Ninety"};
        String ones[] = {"One","Two","Three","Four","Five",
        		"Six","Seven","Eight","Nine","Ten",
        		"Eleven","Twelve","Thirteen","Fourteen","Fifteen"
        		,"Sixteen","Seventeen","Eighteen","Nineteen"};
        String san[] = {"Thousand","Million","Billion"};
        String hun = "Hundred";
        String zero = "Zero";
        if(num==0)
        	return zero;
        int j,k;
        j = 0;
        while(num>0) {
        	String tmp = "";
        	k = num%1000;
        	num = num/1000;
        	if(k>0) {
        		if(j>0)
        			result = san[j-1] + result;
        		if(k>99) {
            		tmp = tmp + " " + ones[k/100-1] + " " + hun;
            		k = k%100;
            	}
            	if(k>19) {
            		tmp = tmp + " " +  tens[k/10-2];
            		k = k%10;
            	}
            	if(k>0)
            		tmp = tmp + " " + ones[k-1];
            	result = tmp + " " + result;	
        	}
        	j++;
        }
        result = result.trim();
        return result;
    }
	
	
	/*137. Single Number II
	 * 数组arrays中每个数字都出现了三次，
	 * 但是有一个数字只出现了一次
	 * 用线性时间，常数空间的算法
	 * 找出只出现一次的数字
	 * */
	public int singleNumber(int[] nums) {
        Arrays.sort(nums);
        int count=0,c=-1,i;
        for(i=0;i<nums.length;i++) {
        	if(count==0) {
        		c = nums[i];
        		count++;
        	}else {
        		if(c!=nums[i])
        			return c;
        		count++;
        		count %= 3;
        	}
        }
        return c;
        /*
        int x1=0,x2=0,mask=0;
        for(int x:nums){
            x2 ^= x1 & c;
            x1 ^= x;
            mask = ~(x1 & x2);
            x2 &= mask;
            x1 &= mask;
        }
        return x1;*/
    }
	
	/*234.回文链表
	 * 判断一个链表是否是回文链表
	 * 即首尾相反
	 * 要求O(N)时间，O(1)空间
	 * 思想是先找中间节点，即两个指针，一个走两步，一个走一步
	 * 
	 * */
	
	public static class ListNode{
		int val;
		ListNode next;
		public ListNode(int x) {
			val = x;
			next = null;
		}
		
		public ListNode() {
			next = null;
		}
	}
	
	public boolean isPalindrome(ListNode head) {
        ListNode p=null,q=null,t=null,s=null;
        
        while(head!=null) {
        	
        	t = head.next;
        	head.next = p;
        	p = head;
        	head = t;
        	if(s!=null && s.val == p.val) {
        		s = s.next;
        	}else{
        		s = p;
        	}
        	if(head==null && q==null)
        		return true;
        	else if(q!=null && head!=null && q.val==head.val) {
        		q = q.next;
        	}else {
        		q = p;
        	}
        }
		return s==null;
    }
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int[] nums = {1,2,1};
//		int []result = nextGreaterElements(nums);
		int i,j;
		String [] A= {"amazon","apple","facebook","google","leetcode"};
		String [] B = {"e","oo"};
		LeetCode test = new LeetCode();
		
		ListNode head=null;
		ListNode t=null,p = null;
		for(int x:nums) {
			p = new ListNode(x);
			p.next = null;
			p.val = x;
			if(head==null) {
				head = p;
				t = p;
			}else {
				t.next = p;
				t = t.next;
			}
		}
		System.out.println(test.isPalindrome(head));
		//System.out.println(test.singleNumber(nums));
		//System.out.println(test.wordSubsets1(A, B));
		//System.out.println(test.numberToWords(1000000));
//		for(i=0;i<result.length;i++)
//			System.out.println(result[i]);
//		System.out.println(intToRoman(4));
//		System.out.println(intToRoman(9));
//		System.out.println(intToRoman(58));
//		System.out.println(intToRoman(1994));
//		List<List<Integer>> result = subsets(nums);
//		for(List<Integer> c:result) {
//			for(int t:c) {
//				System.out.print(t+" ");
//			}
//			System.out.println(";");
//		}
		
		//System.out.println(firstUniqChar("loveleetcode"));
		//System.out.println("loveleetcode".substring(1, 2));
		
		/*List<Integer> result = diffWaysToCompute("2*3-4*5");
		for(int a:result)
			System.out.print(a+" ");*/
//		rotate(nums, 4);
//		for(int c:nums) {
//			System.out.print(c+" ");
//		}
		/*int arr2[][] = {{1,1,1},{1,1,0},{1,0,1},{1,2,1}};
		int result[][] = floodFill(arr2, 1,1,3);
		for(i=0;i<result.length;i++) {
			for(j=0;j<result[0].length;j++)
				System.out.print(result[i][j]+" ");
			System.out.println(";");
		}*/
			
		
	}


}
