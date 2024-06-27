// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using VisitorPatternGenerator;

namespace Nncase.Schedule;

public partial class TileTreeMerger : ITreeNodeVisitor<Unit, bool>
{
    public TileTreeMerger(int opConsumer, int opProducer, int level)
    {
        OpConsumer = opConsumer;
        OpProducer = opProducer;
        Level = level;
    }

    public int OpConsumer { get; }

    public int OpProducer { get; }

    public int Level { get; }

    public bool Merged { get; private set; }

    public bool Visit(ScopeNode value, Unit arg1)
    {
        for (int i = 0; i < value.Children.Count - 1; i++)
        {
            if (value.Children[i] is TileNode producer && value.Children[i + 1] is TileNode consumer &&
                producer.Level == Level && consumer.Level == Level &&
                producer.OpId == OpProducer && consumer.OpId == OpConsumer)
            {
                if (PerformMerge(value, consumer, producer))
                {
                    // simplify structure
                    if (value.Children.Count == 1 && value.Parent is TileNode parent)
                    {
                        var child = value.Children[0];
                        value.Remove(child);
                        parent.Child = child;
                    }

                    return true;
                }
            }
        }

        foreach (var item in value.Children)
        {
            if (item.Accept(this, arg1))
            {
                return true;
            }
        }

        return false;
    }

    public bool Visit(TileNode value, Unit arg1)
    {
        return value.Child.Accept(this, arg1);
    }

    public bool Visit(OpNode value, Unit arg1)
    {
        return false;
    }

    private bool FindOpNode(ITreeNode treeNode, int opId, out OpNode retNode)
    {
        switch (treeNode)
        {
            case OpNode opNode:
                if (opNode.OpId == opId)
                {
                    retNode = opNode;
                    return true;
                }

                break;

            case TileNode tNode:
                return FindOpNode(tNode.Child, opId, out retNode);
            case ScopeNode sNode:
                for (int i = 0; i < sNode.Children.Count; i++)
                {
                    if (FindOpNode(sNode.Children[i], opId, out retNode))
                    {
                        return true;
                    }
                }

                break;
        }

        retNode = null!;
        return false;
    }

    private bool FindFristOpNode(ITreeNode treeNode, out OpNode retNode)
    {
        switch (treeNode)
        {
            case OpNode opNode:
                retNode = opNode;
                return true;
            case TileNode tNode:
                return FindFristOpNode(tNode.Child, out retNode);
            case ScopeNode sNode:
                for (int i = 0; i < sNode.Children.Count; i++)
                {
                    if (FindFristOpNode(sNode.Children[i], out retNode))
                    {
                        return true;
                    }
                }

                break;
        }

        retNode = null!;
        return false;
    }

    private AffineRelation CheckFullMapping(AffineRelation relation, OpNode writeOp)
    {
        var domainVarMap = new Dictionary<int, int>();
        for (int i = 0; i < relation.Domains.Length; i++)
        {
            for (int j = 0; j < relation.Results.Length; j++)
            {
                if (relation.Domains[i] == relation.Results[j])
                {
                    if (!domainVarMap.TryGetValue(i, out _))
                    {
                        domainVarMap.Add(i, j);
                    }
                    else
                    {
                        throw new InvalidOperationException("the same input dim can't equal to muli output dim");
                    }
                }
            }
        }

        return relation;
    }

    private bool PerformMerge(ScopeNode parent, TileNode consumer, TileNode producer)
    {
        if (!FindFristOpNode(consumer, out var firstConsumerOp))
        {
            return false;
        }

        if (firstConsumerOp.Dependences.Count != 1)
        {
            return false;
        }

        // 1. compute the domain realtion : first_consumer_op domain -> producer domain
        var writeOp = firstConsumerOp.Dependences[0].Node;
        var readAccess = firstConsumerOp.Reads[firstConsumerOp.Dependences[0].Index];
        var relation = readAccess * writeOp.Write.Inverse();

        // 2. check the domain rel
        var domainRel = new DomainRelation(firstConsumerOp.OpId, writeOp.OpId, CheckFullMapping(relation, writeOp));

        // 3. compose with merged consumer op's domain realtion.
        if (domainRel.DomainOp != consumer.DomainRelation.RangeOp)
        {
            foreach (var consumerChild in TileTreeWalker.Walk(consumer.Child).OfType<ITileAbleNode>())
            {
                if (consumerChild.DomainRelation.RangeOp == domainRel.RangeOp)
                {
                    domainRel = consumerChild.DomainRelation.ApplyRange(domainRel);
                    break;
                }
            }
        }

        // 4. modify the tree.
        parent.Remove(producer);
        var nextLevelProducer = producer.Child;
        if (consumer.Child is ScopeNode subScope)
        {
            AddProducerToScope(subScope, nextLevelProducer);
        }
        else
        {
            subScope = new ScopeNode();
            AddProducerToScope(subScope, nextLevelProducer);
            subScope.Add(consumer.Child);
            consumer.Child = subScope;
        }

        // when the dd
        if (nextLevelProducer is ScopeNode producerScope)
        {
            // tileAble.DomainRelation = domainRel;
            foreach (var tnode in producerScope.Children.OfType<ITileAbleNode>())
            {
                tnode.DomainRelation = domainRel.ApplyRange(tnode.DomainRelation);
            }
        }
        else if (nextLevelProducer is ITileAbleNode tileAble)
        {
            tileAble.DomainRelation = domainRel;
        }

        return true;
    }

    private void AddProducerToScope(ScopeNode scopeNode, ITreeNode producer)
    {
        if (producer is ScopeNode nextLevelProduceScope)
        {
            scopeNode.InsertRange(0, nextLevelProduceScope.Children);
        }
        else
        {
            scopeNode.Insert(0, producer);
        }
    }
}
