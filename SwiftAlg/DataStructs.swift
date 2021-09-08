//
//  DataStructs.swift
//  SwiftAlg
//
//  Created by yxj on 2021/9/3.
//

import Foundation

public struct Queue<T> {
    fileprivate var array = [T]()
    public var isEmpty: Bool {
        return array.isEmpty
    }
    public var count: Int {
        return array.count
    }
    public mutating func enqueue(_ element: T) {
        array.append(element)
    }
    public mutating func dequeue() -> T? {
        if isEmpty {
           return nil
        } else {
           return array.removeFirst()
        }
     }
    public var front: T? {
        return array.first
    }
}

public struct Stack<T> {
     fileprivate var array = [T]()
     public var isEmpty: Bool {
         return array.isEmpty
     }
     public var count: Int {
        return array.count
     }
     public mutating func push(_ element: T) {
        array.append(element)
     }
     public mutating func pop() -> T? {
        return array.popLast()
     }
     public var top: T? {
        return array.last
     }
}

public struct StackNode {
    var root: TreeNode?
    var visit: Bool = false
    public init (_ root: TreeNode?) {
        self.root = root
    }
}

/**
 用于二叉树的垂序遍历
 */
public struct TreePositionNode {
    var val: Int
    var column: Int = 0
    var row: Int = 0
    public init (_ val: Int, _ column: Int, _ row: Int) {
        self.val = val
        self.column = column
        self.row = row
    }
}

public class ListNode {
    public var val: Int
    public var next: ListNode?
    public init() { self.val = 0; self.next = nil; }
    public init(_ val: Int) { self.val = val; self.next = nil; }
    public init(_ val: Int, _ next: ListNode?) { self.val = val; self.next = next; }
}

public class TreeNode : Equatable {
    public static func == (lhs: TreeNode, rhs: TreeNode) -> Bool {
        lhs === rhs
    }

  public var val: Int
  public var left: TreeNode?
  public var right: TreeNode?
  public init() { self.val = 0; self.left = nil; self.right = nil; }
  public init(_ val: Int) { self.val = val; self.left = nil; self.right = nil; }
  public init(_ val: Int, _ left: TreeNode?, _ right: TreeNode?) {
      self.val = val
      self.left = left
      self.right = right
  }
}

//n叉树的定义
public class Node {
    public var val: Int
    public var children: [Node]
    public init(_ val: Int) {
       self.val = val
       self.children = []
   }
}
class BSTIterator {
    var list: TreeNode? //list 当前指向 next
    func parseBSTToList(_ root: TreeNode?) {
        if root?.right != nil {
            parseBSTToList(root?.right)
        }
        root?.right = list
        list = root
        if root?.left != nil {
            parseBSTToList(root?.left)
        }
    }

    init(_ root: TreeNode?) {
        parseBSTToList(root)
    }

    func next() -> Int {
        let val = list?.val
        list = list?.right
        return val!
    }

    func hasNext() -> Bool {
        if list != nil {
            return true
        } else {
            return false
        }
    }
}

class BSTIterator2 {
    static func inorderTraversal(_ root: TreeNode?) -> [Int] {
        var root = root
        var result : [Int] = []
        var stack : Stack<TreeNode> = Stack()
        while !stack.isEmpty || root != nil{
            while root != nil{
                stack.push(root!)
                root = root?.left
            }
            if !stack.isEmpty {
                let n : TreeNode = stack.pop()!
                result.append(n.val)
                root = n.right
            }
        }
        return result
    }


    let array: [Int]
    var index: Int = 0

    init(_ root: TreeNode?) {
        array = BSTIterator2.inorderTraversal(root)
    }

    func next() -> Int {
        let val = array[index]
        index += 1
        return val
    }

    func hasNext() -> Bool {
        if index+1 <= array.count - 1 {
            return true
        } else {
            return false
        }
    }
}

public struct MinStack {
    var min: Int
    fileprivate var array: [Int]
    fileprivate var minArray: [Int]

    /** initialize your data structure here. */
    init() {
        min = Int.max
        array = []
        minArray = []
    }

    mutating func push(_ val: Int) {
        if val < min {
            min = val
        }
        array.append(val)
        minArray.append(min)
    }

    mutating func pop() {
        array.removeLast()
        minArray.removeLast()
        min = getMin()
    }

    func top() -> Int {
        return array.last ?? 0
    }

    func getMin() -> Int {
        return minArray.last ?? Int.max
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * let obj = MinStack()
 * obj.push(val)
 * obj.pop()
 * let ret_3: Int = obj.top()
 * let ret_4: Int = obj.getMin()
 */

class Codec {
    func serialize(_ root: TreeNode?) -> String {
        var ans: String = ""
        return rserialize(root, &ans)
    }

    func deserialize(_ data: String) -> TreeNode? {
        let arrSubstrings = data.split(separator: ",")
        var arrStrings = arrSubstrings.reversed().compactMap { "\($0)" }
        return rdeserialize(&arrStrings)
    }

    func rserialize(_ root: TreeNode?, _ str: inout String) -> String {
        if root == nil {
            str += "None,"
        } else {
            str = str + String(root!.val) + ","
            str = rserialize(root?.left, &str)
            str = rserialize(root?.right, &str)
        }
        return str
    }

    func rdeserialize(_ strList: inout [String]) -> TreeNode? {
        if strList.last == "None" {
            strList.removeLast()
            return nil
        }
        let root = TreeNode(Int(strList.popLast()!) ?? 0)
        root.left = rdeserialize(&strList)
        root.right = rdeserialize(&strList)
        return root
    }
}

// Your Codec object will be instantiated and called as such:
// var ser = Codec()
// var deser = Codec()
// deser.deserialize(ser.serialize(root))
