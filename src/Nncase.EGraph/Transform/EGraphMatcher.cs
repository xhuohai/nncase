using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.Transform.Pattern;

namespace Nncase.Transform
{
    using EContextEnv = Dictionary<ExprPattern, ENode>;
    using Tuple = IR.Tuple;

    public record EMatchResult(ENode Root, EContextEnv Context)
    {
        public Expr GetExpr(ExprPattern expr)
        {
            return Context[expr].Expr;
        }

        public Var GetExpr(VarPattern pat) => (Var)Context[pat].Expr;
        public Const GetExpr(ConstPattern pat) => (Const)Context[pat].Expr;
        public Function GetExpr(FunctionPattern pat) => (Function)Context[pat].Expr;
        public Call GetExpr(CallPattern pat) => (Call)Context[pat].Expr;
        public Tuple GetExpr(TuplePattern pat) => (Tuple)Context[pat].Expr;


        public T GetExpr<T>(ExprPattern expr) where T : Expr
        {
            return (T)Context[expr].Expr;
        }

        public Expr GetRoot() => Root.Expr;

        public T GetRoot<T>() where T : Expr
        {
            return (T)Root.Expr;
        }

        public (Expr, Expr) GetExpr(ExprPattern pat1, ExprPattern pat2) => (GetExpr(pat1), GetExpr(pat2));
        public (Expr, Expr, Expr) GetExpr(ExprPattern pat1, ExprPattern pat2, ExprPattern pat3) => (GetExpr(pat1), GetExpr(pat2), GetExpr(pat3));
        public (Expr, Expr, Expr, Expr) GetExpr(ExprPattern pat1, ExprPattern pat2, ExprPattern pat3, ExprPattern pat4) => (GetExpr(pat1), GetExpr(pat2), GetExpr(pat3), GetExpr(pat4));

        public (Const, Const) GetExpr(ConstPattern pat1, ConstPattern pat2) => (GetExpr(pat1), GetExpr(pat2));
        public (Const, Const, Const) GetExpr(ConstPattern pat1, ConstPattern pat2, ConstPattern pat3) => (GetExpr(pat1), GetExpr(pat2), GetExpr(pat3));

    }

    public sealed class EGraphMatcher
    {
        public Dictionary<EClass, List<ENode>> eClasses;

        public EGraphMatcher(Dictionary<EClass, List<ENode>> eclasses)
        {
            eClasses = eclasses;
        }

        public (bool, EContextEnv) DefaultMatchEnode(ExprPattern pattern, ENode enode, EContextEnv env)
        {
            throw new NotImplementedException($"Unhandled Match ExprPattern {pattern.GetType()} and Enode {enode.Expr.GetType()} .");
        }


        public (bool, EContextEnv) MatchENode(VArgsPattern Patterns, IEnumerable<EClass> Children, EContextEnv env)
        {
            if (!Patterns.MatchLeaf(Children))
            {
                return (false, env);
            }
            var new_env = env;
            int i = 0;
            foreach (var child in Children)
            {
                var (matchIdx, looped_env) = MatchEclass(Patterns[i], eClasses[child], new_env);
                new_env = looped_env; /* update env */
                if (matchIdx == -1)
                {
                    Patterns.MatchEnd(false);
                    return (false, env);
                }
                i++;
            }
            Patterns.MatchEnd(true);
            return (true, new_env);
        }

        public (bool, EContextEnv) MatchENode(VarPattern pattern, ENode enode, EContextEnv env)
        {
            if (pattern.MatchLeaf((Var)enode.Expr))
            {
                return (true, env);
            }
            return (false, env);
        }

        public (bool, EContextEnv) MatchENode(ConstPattern pattern, ENode enode, EContextEnv env) => (pattern.MatchLeaf((Const)enode.Expr), env);

        public (bool, EContextEnv) MatchENode(FunctionPattern pattern, ENode enode, EContextEnv env)
        {
            var func = (Function)enode.Expr;
            if (pattern.MatchLeaf(func))
            {
                var (matchIdx, new_env) = MatchEclass(pattern.Body, eClasses[enode.Children[0]], env);
                if (matchIdx == -1)
                {
                    return (false, env);
                }
                return MatchENode(pattern.Parameters, enode.Children.Skip(1), env);
            }
            return (false, env);
        }

        public (bool, EContextEnv) MatchENode(CallPattern pattern, ENode enode, EContextEnv env)
        {
            if (pattern.MatchLeaf((Call)enode.Expr))
            {
                var (matchIdx, new_env) = MatchEclass(pattern.Target, eClasses[enode.Children[0]], env);
                if (matchIdx == -1)
                {
                    return (false, env);
                }
                return MatchENode(pattern.Parameters, enode.Children.Skip(1), new_env);
            }
            return (false, env);
        }

        public (bool, EContextEnv) MatchENode(TuplePattern pattern, ENode enode, EContextEnv env)
        {
            if (pattern.MatchLeaf((Tuple)enode.Expr))
            {
                return MatchENode(pattern.Fields, enode.Children, env);
            }
            return (false, env);
        }

        public (bool, EContextEnv) MatchENode(OpPattern pattern, ENode enode, EContextEnv env)
        {
            return (pattern.MatchLeaf((Op)enode.Expr), env);
        }

        private (bool, EContextEnv) UpdateEnv(bool Match, EContextEnv env, ExprPattern pattern, ENode enode)
        {
            if (Match == false)
                return (Match, env);

            if (!env.ContainsKey(pattern))
            {
                var new_env = new EContextEnv(env);
                new_env.Add(pattern, enode);
                return (true, new_env);
            }
            return (env[pattern] == enode, env);
        }

        public (bool, EContextEnv) MatchENode(WildCardPattern pattern, ENode enode, EContextEnv env)
        {
            return (true, env);
        }

        public (bool, EContextEnv) MatchENode(ExprPattern pattern, ENode enode, EContextEnv env)
        {
            var (match, new_env) = (pattern, enode.Expr) switch
            {
                (VarPattern varPat, Var) => MatchENode(varPat, enode, env),
                (ConstPattern conPat, Const) => MatchENode(conPat, enode, env),
                (FunctionPattern functionPat, Function) => MatchENode(functionPat, enode, env),
                (CallPattern callPat, Call) => MatchENode(callPat, enode, env),
                (TuplePattern tuplePat, Tuple) => MatchENode(tuplePat, enode, env),
                (OpPattern opPat, Op) => MatchENode(opPat, enode, env),
                (WildCardPattern wildcardPat, _) => MatchENode(wildcardPat, enode, env),
                (_, _) => (false, env)
            };
            return UpdateEnv(match, new_env, pattern, enode);
        }

        public (int, EContextEnv) MatchEclass(ExprPattern pattern, List<ENode> eNodes, EContextEnv env)
        {
            for (int i = 0; i < eNodes.Count; i++)
            {
                var (match, new_env) = MatchENode(pattern, eNodes[i], env);
                if (match)
                {
                    return (i, new_env);
                }
            }
            return (-1, env);
        }

        public static List<EMatchResult> EMatch(Dictionary<EClass, List<ENode>> eClasses, params ExprPattern[] patterns)
        {
            var matcher = new EGraphMatcher(eClasses);
            var matchResults = new List<EMatchResult>(); // 保存了每个eclassid和入参信息.
            foreach (var pattern in patterns)
            {
                foreach (var (eclass, enodes) in matcher.eClasses)
                {
                    var (matchIdx, env) = matcher.MatchEclass(pattern, enodes, new EContextEnv());
                    if (matchIdx != -1)
                    {
                        matchResults.Add(new EMatchResult(enodes[matchIdx], env));
                    }
                }
            }
            return matchResults;
        }

        public static List<EMatchResult> EMatch(EGraph eGraph, ExprPattern pattern) => EMatch(eGraph.EClasses(), pattern);
    }
}