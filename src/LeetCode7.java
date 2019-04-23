import java.util.Scanner;
import java.io.*;
import java.util.*;
import java.text.*;
import java.math.*;
import java.util.regex.*;

public class LeetCode7 {

	/** 请完成下面这个函数，实现题目要求的功能 **/
	 /** 当然，你也可以不按照这个模板来作答，完全按照自己的想法来 ^-^  **/
	    static String calculate(int m, int k) {
	    	int[] num = new int[m];
	    	int rev[] = new int[m];
	    	if(m==0)
	    		return "";
	    	int i,year = 2019;
	    	for(i=0;i<m;i++) {
	    		if(i==0)
	    			num[i] = 2;
	    		else if(i==1)
	    			num[i] = 3;
	    		else if(i==2)
	    			num[i] = 4;
	    		else
	    			num[i] = num[i-3]+num[i-2];
	    		if(num[i]-1<m)
	    			year++;
	    		rev[i] = revers(num[i]);
	    	}
	    	if(m<4)
	    		year = 2019+m-1;
	    	else year = year-3;
	    	Arrays.sort(rev);
	    	int t = rev[m-k];
	    	t = revers(t);
	    	int s = 0;
	    	for(i=0;i<m;i++)
	    		if(t==num[i]) {
	    			s = i;
	    			break;
	    		}
	    	s++;
	    	return num[m-1]+","+year+","+s+"\n";
	    }
	    
	    		
	    static int revers(int x) {
	    	int res = 0;
	    	while(x>0) {
	    		int t = x%10;
	    		res = res*10+t;
	    		x = x/10;
	    	}
	    	return res;
	    }
	    

	    public static void main(String[] args){
	        Scanner in = new Scanner(System.in);
	        String[] line = in.nextLine().split(",");
	        int m = Integer.valueOf(line[0]);
	        int k = Integer.valueOf(line[1]);
	        System.out.println(calculate(m, k));

	    }
}
