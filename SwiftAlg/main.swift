//
//  main.swift
//  SwiftAlg
//
//  Created by yxj on 2021/3/21.
//

import Foundation

//let s = "(])"
//let d = [")":"(","}":"{","]":"["];
//var arr = [Character]();
//for c in s {
//    if d.values.contains(String(c)) {
//        arr.append(c)
//    } else {
//        if let l = arr.last, let v = d[String(c)]{
//            if v == String(l) {
//                arr.removeLast()
//            } else {
//                print("NO")
//            }
//        } else {
//            print("NO")
//        }
//    }
//}
//if arr.count == 0 {
//    print("YES")
//} else {
//    print("NO")
//}

var a = Solution.init()
//var l = [11, 4, 7, 9, 10, 5, 25]
//let i = a.findKthLargest(&l, 3)
//print(i)

//var node1 = ListNode.init(1)
//var node2 = ListNode.init(2)
//var node3 = ListNode.init(3)
//var node4 = ListNode.init(2)
//var node5 = ListNode.init(1)
//node1.next = node2
//node2.next = node3
//node3.next = node4
//node4.next = node5

//var node = a.reverseList(node1)
//while node != nil {
//    print(node?.val ?? 0)
//    node = node?.next
//}

//var b = a.isPalindrome(node1)
//if b {
//    print(b)
//}

//var arr = [1,1,2,2,3,4,5]
//var cnt = a.removeDuplicates(&arr)
//print(cnt)
//print(arr)

//var s = "-12355"
//var s = " "
//var i = a.myAtoi(s)
//print(i)
//print(Int32.max)
//print(Int32.min)

//var arr = [-1, 1, 2, 3, -4, 6, -1]
//let r = a.maxSubArray(arr)
//print(r)
//[6,2,8,0,4,7,9,null,null,3,5]
var t9 = TreeNode(5)
var t8 = TreeNode(3)
var t7 = TreeNode(9)
var t6 = TreeNode(6)
var t5 = TreeNode(5)
var t4 = TreeNode(4)
var t3 = TreeNode(3, nil, t6)
var t2 = TreeNode(2, t4, t5)
var t1 = TreeNode(1, t2, t3)
//[1,3,2,5,3,null,9]
//var r = a.zigzagLevelOrder(t1)
//print(r)

//print(a.generateTrees(1))

//let root = a.trimBST(t1, 2, 4)
//print(root?.right?.val)
//[6,2,8,0,4,7,9,3,5]
//let node = a.lowestCommonAncestor(t1, TreeNode(2), TreeNode(4))
//let node = a.widthOfBinaryTree(t1)
//let name = readLine()
//let n = a.subsets([1])
//print(n)
//let n = a.longestCommonPrefix(["flower","flow","flight"])
//print(a.strStr("hello", "ll"))
//print(a.plusOne([9]))

//var ser = Codec()
//var deser = Codec()
//let str = ser.serialize(t1)
//let node = deser.deserialize(str)
//print(str)
/*
 [1,2,3,0,0,0]
 3
 [2,5,6]
 3
 */
//var num1 = [2,0]
//a.merge(&num1, 1, [1], 1)
//let ans = a.isPalindrome("A man, a plan, a canal: Panama")
//let ans = a.climbStairs(44)
//print(ans)

//var ht:MyHashTable<Int, String> = MyHashTable(10)
//ht.setValue("hello", forKey: 1)
//ht.setValue("world", forKey: 2)
//ht.removeValue(forKey: 2)
//print(ht.value(forKey: 1))



//let sorter = Sorter()
//let ans = sorter.sortArray([5, 4, 3, 6, 1, 2])
//print(ans)

let res = a.findDisappearedNumbers([2,2])//([-2,-1,-1,1,1,2,2], 0)//([1,3,-1,-3,5,3,6,7], 3)
//let res = a.minWindow("ba", "a")
print(res)

