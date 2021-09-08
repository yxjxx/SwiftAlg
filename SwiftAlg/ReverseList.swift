//
//  ReverseList.swift
//  SwiftAlg
//
//  Created by yxj on 2021/9/8.
//

import Foundation
/*
 包含：
 1. 反转链表
 2. 反转链表前k个结点
 3. 反转链表中的部分 m...n
 4. 每 k 个一组反转链表
 */
class ReverseList {
    func isCanReverse(_ head: ListNode?, _ k: Int) -> Bool {
        var current = head
        var count = 0
        while current != nil {
            count += 1
            if count >= k {
                return true
            }
            current = current?.next
        }
        return false
    }

    func reverseKGroup(_ head: ListNode?, _ k: Int) -> ListNode? {
        var result: ListNode? = head
        var current = head
        let canReverse = isCanReverse(current, k)
        if canReverse {
            let tail = current
            (result, current) = reverseList(current, count: k)
            let sublistHead = reverseKGroup(current, k)
            tail?.next = sublistHead
        }

        return result
    }

    func reverseBetween(_ head: ListNode?, _ left: Int, _ right: Int) -> ListNode? {
        if right < left {
            return head
        }
        var enterPre: ListNode?
        var enter: ListNode?
        var tailNext: ListNode?
        var subList: ListNode?
        var current = head
        var index:Int = 1
        while current != nil {
            if index == left - 1 {
                enterPre = current
                current = current?.next
                index += 1
            } else if index == left {
                enter = current
                (subList, tailNext) = reverseList(current, count: right - left + 1)
                enterPre?.next = subList
                enter?.next = tailNext
                break
            } else {
                current = current?.next
                index += 1
            }
        }

        if left == 1 {
            return subList
        }
        return head
    }

    /**
     反转链表前 count 个节点
     返回值为（新链表的头节点，子链表之后的一个节点也即原链表第 count+1 个节点）
     一个可能有用的信息：反转后的子链表的最后一个节点就是传进来的 head
     */
    func reverseList(_ head: ListNode?, count: Int) -> (ListNode?, ListNode?) {
        var currentNode = head
        var newNode : ListNode?
        var tempNode: ListNode?
        var times = 0
        while (currentNode != nil && times < count) {
            tempNode = currentNode?.next
            currentNode?.next = newNode
            newNode = currentNode
            currentNode = tempNode
            times += 1
        }
        return (newNode, tempNode)
    }

    func generateLinkedList(count: Int) -> ListNode {
        let head = ListNode(1)
        var current = head
        for i in Array(2...count) {
            let node = ListNode(i)
            current.next = node
            current = node
        }
        return head
    }

    func printLinkedList(_ header: ListNode?) {
        var node = header
        while node != nil {
            print(node!.val)
            node = node!.next
        }
    }

    func linkedListCount(_ head: ListNode?) -> Int {
        var current = head
        var count = 0
        while current != nil {
            count += 1
            current = current?.next
        }
        return count
    }
}
