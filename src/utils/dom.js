export function createCharacterItemHTML(name, avatar, score = null) {
    const avatarUrl = getThumbnailUrl('avatar', avatar);
    const scoreHtml = score !== null ? `<div class="charSim-score">${score.toFixed(4)}</div>` : '';
    
    return `
        <div class="charSim-character-item" data-avatar="${avatar}">
            <img src="${avatarUrl}" alt="${name}'s avatar">
            <span class="charSim-name">${name}</span>
            ${scoreHtml}
        </div>
    `;
}

export function createClusterGroupHTML(clusterData) {
    const membersHtml = clusterData.members.map(member => 
        createCharacterItemHTML(member.name, member.avatar, member.intraClusterDistance)
    ).join('');

    return `
        <div class="charSim-cluster-group">
            <div class="charSim-cluster-header">
                <span>Cluster Uniqueness:</span>
                <span class="charSim-score">${clusterData.clusterUniqueness.toFixed(4)}</span>
            </div>
            ${membersHtml}
        </div>
    `;
}