import { calculateMeanEmbedding, cosineSimilarity, normalizeSimilarity, manhattanDistance } from "../utils/math.js";

export class AnalysisEngine {
    constructor(embeddingManager) {
        this.manager = embeddingManager;
    }

    calculateUniqueness() {
        const embeddings = Array.from(this.manager.getEmbeddings().values());
        const libraryMean = calculateMeanEmbedding(embeddings);
        
        if (!libraryMean) return [];

        const dimension = libraryMean.length;
        const results = [];

        for (const [avatar, embedding] of this.manager.getEmbeddings().entries()) {
            const distance = manhattanDistance(embedding, libraryMean);
            const char = this.manager.getCharacter(avatar);
            if (char) {
                results.push({ avatar, name: char.name, distance });
            }
        }
        return results;
    }

    async calculateClusters(threshold) {
        const embeddings = this.manager.getEmbeddings();
        if (embeddings.size === 0) return [];

        return new Promise((resolve, reject) => {
            // We use the global 'cluster' function from lib/cluster.js
            if (typeof cluster === 'undefined') {
                reject("Clustering library not loaded.");
                return;
            }

            const workerCode = `
                const e={52:e=>{function n(e,n){var t=e.nodes.slice(),r=[];if(!t.length)return r;for(var o=[],i=null;t.length;)for(o.length||(i&&r.push(i),i={nodes:(o=[t.pop()]).slice(),edges:{}});o.length;)for(var d=o.pop(),a=t.length-1;a>=0;a--){var s=t[a];if(e.edges[d.id]&&e.edges[d.id][s.id]>=n){o.push(s),i.nodes.push(s),i.edges[d.id]=i.edges[d.id]||{},i.edges[d.id][s.id]=e.edges[d.id][s.id],i.edges[s.id]=i.edges[s.id]||{},i.edges[s.id][d.id]=e.edges[s.id][d.id];var c=t.slice(0,a).concat(t.slice(a+1));t=c}}return i&&r.push(i),r}e.exports={create:function(e,n){var t={},r=1,o=e.map((function(e){return{id:r++,data:e}}));return o.forEach((function(e){t[e.id]=t[e.id]||{},o.forEach((function(r){if(e!==r){var o=n(e.data,r.data);t[e.id][r.id]=o,t[r.id]=t[r.id]||{},t[r.id][e.id]=o}}))})),{nodes:o,edges:t}},data:function(e){return e.data},connected:n,divide:function(e,t,r){for(var o=1/0,i=0,d=function(e){var n=0;return e.nodes.forEach((function(t){e.nodes.forEach((function(r){var o=e.edges[t.id][r.id];o&&o>n&&(n=o)}))})),n}(e)+1,a=null,s=-2;s<r;s++){var c,u=n(e,c=-2==s?i:-1==s?d:(d+i)/2),f=u.length-t;if(f<o&&f>=0&&(o=f,a=u),u.length>t&&(d=c),u.length<t&&(i=c),u.length==t)break;if(i==d)break}return a},findCenter:function(e){var n=function(e){var n={};return e.nodes.forEach((function(e){n[e.id]={},n[e.id][e.id]=0})),e.nodes.forEach((function(t){e.nodes.forEach((function(r){if(t!=r){var o=e.edges[t.id]&&e.edges[t.id][r.id];null==o&&(o=1/0),n[t.id][r.id]=o}}))})),e.nodes.forEach((function(t){e.nodes.forEach((function(r){e.nodes.forEach((function(e){var o=n[r.id][t.id]+n[t.id][e.id];n[r.id][e.id]>o&&(n[r.id][e.id]=o)}))}))})),n}(e),t=1/0,r=null;return e.nodes.forEach((function(o){var i=0;e.nodes.forEach((function(e){var t=n[o.id][e.id];t>i&&(i=t)})),t>i&&(t=i,r=o)})),r},growFromNuclei:function(e,n){for(var t=n.map((function(e){return{nodes:[e],edges:{}}})),r=e.nodes.filter((function(e){return 0==n.filter((function(n){return e==n})).length})),o=0,i=t.length;r.length&&i;){i-=1;var d=t[o];o=(o+1)%t.length;var a=null,s=null,c=-1/0;if(d.nodes.forEach((function(n){r.forEach((function(t){var r=e.edges[n.id]&&e.edges[n.id][t.id];r&&r>c&&(a=n,s=t,c=r)}))})),a){var u=a,f=s;d.edges[u.id]=d.edges[u.id]||{},d.edges[u.id][f.id]=e.edges[u.id][f.id],d.edges[f.id]=d.edges[f.id]||{},d.edges[f.id][u.id]=e.edges[f.id][u.id],d.nodes.push(s),r=r.filter((function(e){return e!=s})),i=t.length}}return{graphs:t,orphans:r}}}},834:(e,n,t)=>{var r=t(52);e.exports=function(e,n){var t,o=r.create(e,(function(e,t){var r=n(e,t);if("number"!=typeof r||r<0)throw new Error("Similarity function did not yield a number in the range [0, +Inf) when comparing "+e+" to "+t+" : "+r);return r}));function i(e){return function(){return e.apply(this,Array.prototype.slice.call(arguments)).map((function(e){return e.nodes.map(r.data)}))}}function d(e,n){var t=n||1e3;return r.divide(o,e,t)}function a(e,n){var t=d(e,n);return t.sort((function(e,n){return n.nodes.length-e.nodes.length})),t.splice(e),t.map(r.findCenter)}return{groups:i(d),representatives:(t=a,function(){return t.apply(this,Array.prototype.slice.call(arguments)).map(r.data)}),similarGroups:i((function(e){return r.connected(o,e)})),evenGroups:function(e,n){for(var t=a(e),i=r.growFromNuclei(o,t),d=i.graphs.map((function(e){return e.nodes.map(r.data)}));i.orphans.length;){var s=r.data(i.orphans.pop());d.sort((function(e,n){return e.length-n.length})),d[0].push(s)}return d}}}}},n={};function t(r){var o=n[r];if(void 0!==o)return o.exports;var i=n[r]={exports:{}};return e[r](i,i.exports,t),i.exports}var cluster = t(834);
                self.onmessage = function(event) {
                    const { embeddings, threshold } = event.data;
                    const data = Array.from(embeddings.entries()).map(([avatar, embedding]) => ({ avatar, embedding }));
                    const clustering = cluster(data, (a, b) => {
                        const similarity = cosineSimilarity(a.embedding, b.embedding);
                        return normalizeSimilarity(similarity);
                    });
                    const groups = clustering.similarGroups(threshold);
                    self.postMessage(groups);
                };
            `;

            const blob = new Blob([workerCode], { type: 'application/javascript' });
            const worker = new Worker(URL.createObjectURL(blob));

            worker.onmessage = (e) => {
                const rawGroups = e.data;
                const libraryMean = calculateMeanEmbedding(Array.from(embeddings.values()));
                
                const processedGroups = rawGroups.map(group => {
                    const memberEmbeddings = group.map(m => embeddings.get(m.avatar));
                    const clusterMean = calculateMeanEmbedding(memberEmbeddings);
                    
                    const clusterUniqueness = clusterMean ? manhattanDistance(clusterMean, libraryMean) : 0;

                    const members = group.map(m => {
                        const char = this.manager.getCharacter(m.avatar);
                        const emb = embeddings.get(m.avatar);
                        const intraDist = clusterMean ? manhattanDistance(emb, clusterMean) : 0;
                        return { 
                            avatar: m.avatar, 
                            name: char ? char.name : "Unknown", 
                            intraClusterDistance: intraDist 
                        };
                    });

                    return { clusterUniqueness, members };
                });

                resolve(processedGroups);
                worker.terminate();
            };

            worker.onerror = (err) => {
                reject(err);
                worker.terminate();
            };

            worker.postMessage({ embeddings, threshold });
        });
    }
}